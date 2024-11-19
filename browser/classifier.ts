import * as tf from '@tensorflow/tfjs'
import { ImageModel, cachedLoadLayersModel } from './model'
import {
  ClassificationResult,
  calcClassWeight,
  createImageClassifier,
  getClassCount,
  mapWithClassName,
} from '../classifier-utils'
import { disposeTensor, toOneTensor } from '../tensor'

export type ClassifierModel = Awaited<
  ReturnType<typeof loadImageClassifierModel>
>

export async function loadImageClassifierModel(options: {
  baseModel: ImageModel
  hiddenLayers?: number[]
  modelUrl?: string
  cacheUrl?: string
  checkForUpdates?: boolean
  /** @description if not provided, will be auto scanned from datasetDir or load from the model.json */
  classNames?: string[]
}) {
  let { baseModel, classNames } = options

  async function loadClassifierModel() {
    let { modelUrl: url, cacheUrl, checkForUpdates } = options
    if (url && cacheUrl) {
      try {
        let model = await cachedLoadLayersModel({
          url,
          cacheUrl,
          checkForUpdates,
          classNames,
        })
        return model
      } catch (error) {
        if (!String(error).includes('file not found')) {
          throw error
        }
      }
    }
    return createImageClassifier({
      embeddingFeatures: baseModel.spec.features,
      hiddenLayers: options.hiddenLayers,
      get classes() {
        if (!classNames) {
          throw new Error('classNames not provided')
        }
        return classNames.length
      },
      classNames,
    })
  }

  let classifierModel = await loadClassifierModel()
  classNames = classifierModel.classNames
  if (!classNames) {
    throw new Error('classNames not provided')
  }
  let classCount = classNames.length
  if (classCount < 2) {
    throw new Error('expect at least 2 classes')
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

  async function classifyImageUrl(
    url: string,
  ): Promise<ClassificationResult[]> {
    let embedding = await baseModel.imageUrlToEmbedding(url)
    /* do not dispose embedding because it may be cached */
    return classifyImageEmbedding(embedding)
  }

  async function classifyImageFile(
    file: File,
  ): Promise<ClassificationResult[]> {
    let embedding = await baseModel.imageFileToEmbedding(file)
    /* do not dispose embedding because it may be cached */
    return classifyImageEmbedding(embedding)
  }

  async function classifyImage(
    image: Parameters<typeof tf.browser.fromPixels>[0],
  ): Promise<ClassificationResult[]> {
    let imageTensor = await tf.browser.fromPixelsAsync(image)
    let embedding = baseModel.imageTensorToEmbedding(imageTensor)
    imageTensor.dispose()
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
    return mapWithClassName(classNames!, values)
  }

  async function train(
    options: tf.ModelFitArgs & {
      x: tf.Tensor<tf.Rank>
      y: tf.Tensor<tf.Rank>
      /** @description to calculate classWeight */
      classCounts?: number[]
    },
  ): Promise<tf.History> {
    if (!compiled) {
      compile()
    }
    let { x, y, classCounts, ...rest } = options
    let classWeight =
      options.classWeight ||
      (classCounts
        ? calcClassWeight({
            classes: classCount,
            classCounts,
          })
        : undefined)
    let history = await classifierModel.fit(x, y, {
      ...options,
      shuffle: true,
      classWeight,
    })
    return history
  }

  return {
    baseModel,
    classifierModel,
    classNames,
    classifyImageUrl,
    classifyImageFile,
    classifyImageTensor,
    classifyImage,
    classifyImageEmbedding,
    compile,
    train,
  }
}
