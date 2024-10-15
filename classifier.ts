import { ImageModel, loadLayersModel, saveModel } from './model'
import * as tf from '@tensorflow/tfjs-node'
import { existsSync } from 'fs'
import { join } from 'path'
import { disposeTensor, toOneTensor } from './tensor'
import { ClassWeight, ClassWeightMap } from '@tensorflow/tfjs-node'
import { startTimer } from '@beenotung/tslib/timer'
import { getDirFilenames, getDirFilenamesSync } from '@beenotung/tslib/fs'

export type ClassifierModelSpec = {
  embeddingFeatures: number
  hiddenLayers?: number[]
  classes: number
}

export function createImageClassifier(spec: ClassifierModelSpec) {
  let { hiddenLayers } = spec

  if (spec.classes < 2) {
    throw new Error('image classifier must be at least 2 classes')
  }

  let classifierModel = tf.sequential()
  classifierModel.add(
    tf.layers.inputLayer({ inputShape: [spec.embeddingFeatures] }),
  )
  classifierModel.add(tf.layers.dropout({ rate: 0.2 }))
  if (hiddenLayers) {
    for (let i = 0; i < hiddenLayers.length; i++) {
      classifierModel.add(
        tf.layers.dense({ units: hiddenLayers[i], activation: 'gelu' }),
      )
    }
  }
  classifierModel.add(
    tf.layers.dense({ units: spec.classes, activation: 'softmax' }),
  )

  return classifierModel
}

export type ClassificationResult = {
  label: string
  /** @description between 0 to 1 */
  confidence: number
}

export type ClassifierModel = Awaited<
  ReturnType<typeof loadImageClassifierModel>
>

export async function loadImageClassifierModel(options: {
  baseModel: ImageModel
  hiddenLayers?: number[]
  modelDir: string
  datasetDir: string
  classNames?: string[]
}) {
  let { baseModel, datasetDir, modelDir } = options

  let classNames = options.classNames || getDirFilenamesSync(datasetDir)

  if (classNames.length < 2) {
    throw new Error('expect at least 2 classes')
  }

  let classifierModel = existsSync(modelDir)
    ? await loadLayersModel({ dir: modelDir })
    : createImageClassifier({
        embeddingFeatures: baseModel.spec.features,
        hiddenLayers: options.hiddenLayers,
        classes: classNames.length,
      })

  let classCount = getClassCount(classifierModel.outputShape)
  if (classCount != classNames.length) {
    throw new Error(
      'number of classes mismatch, expected: ' +
        classNames.length +
        ', got: ' +
        classCount,
    )
  }

  let compiled = false

  function compile() {
    compiled = true
    classifierModel.compile({
      optimizer: 'adam',
      loss: tf.metrics.categoricalCrossentropy,
      metrics: [tf.metrics.categoricalAccuracy],
    })
  }

  async function classifyImageFile(
    file: string,
  ): Promise<ClassificationResult[]> {
    let embedding = await baseModel.imageFileToEmbedding(file)
    /* do not dispose embedding because it may be cached */
    return classifyImageEmbedding(embedding)
  }

  async function classifyImageTensor(
    imageTensor: tf.Tensor3D | tf.Tensor4D,
  ): Promise<ClassificationResult[]> {
    let embedding = baseModel.imageTensorToEmbedding(imageTensor)
    let results = await classifyImageEmbedding(embedding)
    embedding.dispose()
    return results
  }

  async function classifyImageEmbedding(embedding: tf.Tensor) {
    let outputs = tf.tidy(() => {
      let outputs = classifierModel.predict(embedding)
      return toOneTensor(outputs)
    })
    let values = await outputs.data()
    disposeTensor(outputs)
    return mapWithClassName(classNames, values)
  }

  async function loadDatasetFromDirectory() {
    let xs: tf.Tensor[] = []
    let classIndices: number[] = []
    let classCounts: number[] = new Array(classNames.length).fill(0)

    let total = 0
    let classes = await Promise.all(
      classNames.map(async (className, classIdx) => {
        let dir = join(datasetDir, className)
        let filenames = await getDirFilenames(dir)
        total += filenames.length
        return { classIdx, dir, filenames }
      }),
    )

    let timer = startTimer('load dataset')
    timer.setEstimateProgress(total)
    for (let { classIdx, dir, filenames } of classes) {
      for (let filename of filenames) {
        let file = join(dir, filename)
        let embedding = await baseModel.imageFileToEmbedding(file)
        xs.push(embedding)
        classIndices.push(classIdx)
        classCounts[classIdx]++
        timer.tick()
      }
    }
    timer.end()

    let x = tf.concat(xs)

    if (!baseModel.fileEmbeddingCache) {
      for (let x of xs) {
        x.dispose()
      }
    }

    let y = tf.tidy(() =>
      tf.oneHot(tf.tensor1d(classIndices, 'int32'), classNames.length),
    )

    return { x, y, classCounts }
  }

  async function train(
    options?: tf.ModelFitArgs &
      (
        | {
            x: tf.Tensor<tf.Rank>
            y: tf.Tensor<tf.Rank>
            /** @description to calculate classWeight */
            classCounts?: number[]
          }
        | {}
      ),
  ): Promise<tf.History> {
    if (!compiled) {
      compile()
    }
    let next = async (
      x: tf.Tensor<tf.Rank>,
      y: tf.Tensor<tf.Rank>,
      classWeight?: ClassWeight | ClassWeight[] | ClassWeightMap,
    ) => {
      let history = await classifierModel.fit(x, y, {
        ...options,
        shuffle: true,
        classWeight,
      })
      return history
    }
    if (options && 'x' in options && 'y' in options) {
      let { x, y, classCounts, ...rest } = options
      options = rest
      let classWeight =
        options.classWeight ||
        (classCounts
          ? calcClassWeight({
              classes: classNames.length,
              classCounts,
            })
          : undefined)
      return next(x, y, classWeight)
    } else {
      let { x, y, classCounts } = await loadDatasetFromDirectory()
      let classWeight =
        options?.classWeight ||
        calcClassWeight({
          classes: classNames.length,
          classCounts,
        })
      let history = await next(x, y, classWeight)
      x.dispose()
      y.dispose()
      return history
    }
  }

  async function save(dir = modelDir) {
    return await saveModel({
      model: classifierModel,
      dir,
    })
  }

  return {
    baseModel,
    classifierModel,
    classNames,
    classifyImageFile,
    classifyImageTensor,
    classifyImageEmbedding,
    loadDatasetFromDirectory,
    compile,
    train,
    save,
  }
}

function getClassCount(shape: tf.Shape | tf.Shape[]): number {
  for (;;) {
    let value = shape[0]
    for (let i = 1; i < shape.length; i++) {
      value = shape[i] || value
    }
    if (Array.isArray(value)) {
      shape = value
      continue
    }
    if (!value) {
      throw new Error('failed to get class count')
    }
    return value
  }
}

export function topClassifyResult(
  items: ClassificationResult[],
): ClassificationResult {
  let idx = 0
  let max = items[idx]
  for (let i = 1; i < items.length; i++) {
    let item = items[i]
    if (item.confidence > max.confidence) {
      max = item
    }
  }
  return max
}

/**
 * @description the values is returned as is.
 * It should has be applied softmax already.
 * */
export function mapWithClassName(
  classNames: string[],
  values: ArrayLike<number>,
  options?: {
    sort?: boolean
  },
): ClassificationResult[] {
  let result = new Array(classNames.length)
  for (let i = 0; i < classNames.length; i++) {
    result[i] = {
      label: classNames[i],
      confidence: values[i],
    }
  }
  if (options?.sort) {
    result.sort((a, b) => b.confidence - a.confidence)
  }
  return result
}

export function calcClassWeight(options: {
  classes: number
  classCounts: number[]
}) {
  let total = options.classCounts.reduce((acc, c) => acc + c, 0)
  let classWeights = options.classCounts.map(
    count => total / options.classes / count,
  )
  return classWeights
}
