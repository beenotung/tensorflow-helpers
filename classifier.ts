import { ImageModel, loadLayersModel, saveModel } from './model'
import * as tf from '@tensorflow/tfjs'
import { existsSync, readdirSync, statSync } from 'fs'
import { join } from 'path'
import { ClassWeight, ClassWeightMap } from '@tensorflow/tfjs'
import { startTimer } from '@beenotung/tslib/timer'
import { getDirFilenames, getDirFilenamesSync } from '@beenotung/tslib/fs'
import {
  ClassificationResult,
  calcClassWeight,
  createImageClassifier,
  getClassCount,
  mapWithClassName,
} from './classifier-utils'
export * from './classifier-utils'

export type ClassifierModel = Awaited<
  ReturnType<typeof loadImageClassifierModel>
>

export async function loadImageClassifierModel(options: {
  baseModel: ImageModel
  hiddenLayers?: number[]
  modelDir: string
  datasetDir: string
  /** @description if not provided, will be auto scanned from datasetDir or load from the model.json */
  classNames?: string[]
}) {
  let { baseModel, datasetDir, modelDir, classNames } = options

  let classifierModel = existsSync(modelDir)
    ? await loadLayersModel({ dir: modelDir, classNames })
    : createImageClassifier({
        embeddingFeatures: baseModel.spec.features,
        hiddenLayers: options.hiddenLayers,
        get classes() {
          if (!classNames) {
            return getClassesFromDatasetDir(datasetDir).length
          }
          return classNames.length
        },
        classNames,
      })
  classNames = classifierModel.classNames || classNames
  if (!classNames && existsSync(datasetDir)) {
    classNames = getClassesFromDatasetDir(datasetDir)
  }
  if (!classNames) {
    throw new Error('classNames not provided')
  }
  if (classNames.length < 2) {
    throw new Error('expect at least 2 classes')
  }
  let classCount = getClassCount(classifierModel.outputShape)
  if (classNames.length !== classCount) {
    throw new Error(
      `classNames length mismatch: given ${classNames.length} classes, but the model has ${classCount} outputs`,
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
      if (embedding.rank === 1) {
        embedding = tf.expandDims(embedding, 0)
      }
      let y = classifierModel.predict(embedding) as tf.Tensor
      let probs = tf.squeeze(y, [0])
      return probs
    })
    let values = await outputs.data()
    outputs.dispose()
    return mapWithClassName(classNames!, values)
  }

  async function loadDatasetFromDirectory() {
    let xs: tf.Tensor[] = []
    let classIndices: number[] = []
    let classCounts: number[] = new Array(classCount).fill(0)

    let total = 0
    let classes = []
    let timer = startTimer('scan dataset')
    timer.setEstimateProgress(classNames!.length)
    for (let i = 0; i < classNames!.length; i++) {
      let className = classNames![i]
      let dir = join(datasetDir, className)
      let filenames = await getDirFilenames(dir)
      total += filenames.length
      classes.push({
        classIdx: i,
        dir,
        filenames,
      })
      timer.tick()
    }

    timer.next('load dataset')
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

    timer.next('stack embeddings')
    let x = tf.concat(xs)

    if (!baseModel.fileEmbeddingCache) {
      timer.next('dispose individual embeddings')
      timer.setEstimateProgress(xs.length)
      for (let x of xs) {
        x.dispose()
        timer.tick()
      }
    }

    timer.next('prepare one-hot labels')
    let y = tf.tidy(() =>
      tf.oneHot(tf.tensor1d(classIndices, 'int32'), classCount),
    )
    timer.end()

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
              classes: classCount,
              classCounts,
            })
          : undefined)
      return next(x, y, classWeight)
    } else {
      let { x, y, classCounts } = await loadDatasetFromDirectory()
      let classWeight =
        options?.classWeight ||
        calcClassWeight({
          classes: classCount,
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
      classNames,
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

export function getClassesFromDatasetDir(datasetDir: string) {
  if (!existsSync(datasetDir)) {
    throw new Error('datasetDir not exists')
  }
  let classNames = getDirFilenamesSync(datasetDir).filter(filename => {
    let file = join(datasetDir, filename)
    return statSync(file).isDirectory()
  })
  if (classNames.length === 0) {
    throw new Error('no image folders found in datasetDir')
  }
  return classNames
}
