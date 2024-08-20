import { ImageModel, loadLayersModel, saveModel } from './model'
import * as tf from '@tensorflow/tfjs-node'
import { readdirSync, existsSync } from 'fs'
import { readdir } from 'fs/promises'
import { join } from 'path'
import { disposeTensor, toOneTensor } from './tensor'
import { ClassWeight, ClassWeightMap } from '@tensorflow/tfjs-node'

export type ClassifierModelSpec = {
  embeddingFeatures: number
  hiddenLayers?: number[]
  classes: number
}

export function createImageClassifier(spec: ClassifierModelSpec) {
  let { hiddenLayers } = spec

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

  let classNames = options.classNames || readdirSync(datasetDir)

  let classifierModel = existsSync(modelDir)
    ? await loadLayersModel({ dir: modelDir })
    : createImageClassifier({
        embeddingFeatures: baseModel.spec.features,
        hiddenLayers: options.hiddenLayers,
        classes: classNames.length,
      })

  let compiled = false

  function compile() {
    compiled = true
    classifierModel.compile({
      optimizer: 'adam',
      loss: tf.metrics.categoricalCrossentropy,
      metrics: [tf.metrics.categoricalAccuracy],
    })
  }

  async function classifyAsync(
    file_or_image_tensor: string | tf.Tensor,
  ): Promise<ClassificationResult[]> {
    let embedding = await baseModel.inferEmbeddingAsync(file_or_image_tensor)
    let outputs = classifierModel.predict(embedding)
    embedding.dispose()
    let values = await toOneTensor(outputs).data()
    disposeTensor(outputs)
    return mapWithClassName(classNames, values)
  }

  function classifySync(
    file_or_image_tensor: string | tf.Tensor,
  ): ClassificationResult[] {
    let embedding = baseModel.inferEmbeddingSync(file_or_image_tensor)
    let outputs = classifierModel.predict(embedding)
    embedding.dispose()
    let values = toOneTensor(outputs).dataSync()
    disposeTensor(outputs)
    return mapWithClassName(classNames, values)
  }

  async function loadDatasetFromDirectoryAsync(
    options: { cache?: boolean } = {},
  ) {
    let xs: tf.Tensor[] = []
    let classIndices: number[] = []
    let classCounts: number[] = new Array(classNames.length).fill(0)

    let total = 0
    let classes = await Promise.all(
      classNames.map(async (className, classIdx) => {
        let dir = join(datasetDir, className)
        let filenames = await readdir(dir)
        total += filenames.length
        return { classIdx, dir, filenames }
      }),
    )

    let i = 0
    let nextI = 0
    for (let { classIdx, dir, filenames } of classes) {
      for (let filename of filenames) {
        i++
        if (i >= nextI) {
          process.stderr.write(`\rload dataset: ${i}/${total}`)
          nextI += total / 100
        }
        let file = join(dir, filename)
        let embedding = await baseModel.inferEmbeddingAsync(file, options)
        xs.push(embedding)
        classIndices.push(classIdx)
        classCounts[classIdx]++
      }
    }
    process.stderr.write(`\rload dataset: ${i}/${total}\n`)

    let x = tf.concat(xs)

    if (!options.cache) {
      for (let x of xs) {
        x.dispose()
      }
    }

    let y = tf.tidy(() =>
      tf.oneHot(tf.tensor1d(classIndices, 'int32'), classNames.length),
    )

    return { x, y, classCounts }
  }

  async function trainAsync(
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
      let { x, y, classCounts } = await loadDatasetFromDirectoryAsync()
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
    classifyAsync,
    classifySync,
    loadDatasetFromDirectoryAsync,
    compile,
    trainAsync,
    save,
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
