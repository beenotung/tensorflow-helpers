import { ImageModel, loadLayersModel, saveModel } from './model'
import * as tf from '@tensorflow/tfjs-node'
import { readdirSync, existsSync } from 'fs'
import { readdir } from 'fs/promises'
import { join } from 'path'
import { disposeTensor, toOneTensor } from './tensor'

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

  async function loadDatasetFromDirectoryAsync() {
    let xs: tf.Tensor[] = []
    let class_indices: number[] = []
    for (let class_idx = 0; class_idx < classNames.length; class_idx++) {
      let dir = join(datasetDir, classNames[class_idx])
      let filenames = await readdir(dir)
      for (let filename of filenames) {
        let file = join(dir, filename)
        let embedding = await baseModel.inferEmbeddingAsync(file)
        xs.push(embedding)
        class_indices.push(class_idx)
      }
    }

    let x = tf.concat(xs)
    for (let x of xs) {
      x.dispose()
    }

    let class_indices_tensor = tf.tensor1d(class_indices, 'int32')
    let y = tf.oneHot(class_indices_tensor, classNames.length)
    class_indices_tensor.dispose()

    return { x, y }
  }

  async function trainAsync(
    options?: tf.ModelFitArgs &
      (
        | {
            x: tf.Tensor<tf.Rank>
            y: tf.Tensor<tf.Rank>
          }
        | {}
      ),
  ): Promise<tf.History> {
    if (!compiled) {
      compile()
    }
    var x: tf.Tensor<tf.Rank>
    var y: tf.Tensor<tf.Rank>
    let loadedDataset = false
    if (options && 'x' in options && 'y' in options) {
      var { x, y, ...rest } = options
      options = rest
    } else {
      var { x, y } = await loadDatasetFromDirectoryAsync()
      loadedDataset = true
    }
    let history = await classifierModel.fit(x, y, {
      ...options,
      shuffle: true,
    })
    if (loadedDataset) {
      x.dispose()
      y.dispose()
    }
    return history
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
