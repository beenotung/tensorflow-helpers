import { ImageModel, loadLayersModel, saveModel } from './model'
import * as tf from '@tensorflow/tfjs-node'
import { readdirSync, existsSync } from 'fs'
import { readdir } from 'fs/promises'
import { join } from 'path'
import { disposeTensor, toOneTensor } from './tensor'

export type ClassifierModelSpec = {
  embedding_features: number
  hidden_layers?: number[]
  classes: number
}

export function createImageClassifier(spec: ClassifierModelSpec) {
  let { hidden_layers } = spec

  let classifierModel = tf.sequential()
  classifierModel.add(
    tf.layers.inputLayer({ inputShape: [spec.embedding_features] }),
  )
  classifierModel.add(tf.layers.dropout({ rate: 0.2 }))
  if (hidden_layers) {
    for (let i = 0; i < hidden_layers.length; i++) {
      classifierModel.add(
        tf.layers.dense({ units: hidden_layers[i], activation: 'gelu' }),
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
  /**
   * @description between 0 to 1. Also called probability or confidence
   */
  score: number
}

export type ClassifierModel = Awaited<
  ReturnType<typeof loadImageClassifierModel>
>

export async function loadImageClassifierModel(options: {
  baseModel: ImageModel
  hidden_layers?: number[]
  modelDir: string
  datasetDir: string
  class_names?: string[]
}) {
  let { baseModel, datasetDir, modelDir } = options

  let class_names = options.class_names || readdirSync(datasetDir)

  let classifierModel = existsSync(modelDir)
    ? await loadLayersModel({ dir: modelDir })
    : createImageClassifier({
        embedding_features: baseModel.spec.features,
        hidden_layers: options.hidden_layers,
        classes: class_names.length,
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
    return mapWithClassName(values)
  }

  function classifySync(
    file_or_image_tensor: string | tf.Tensor,
  ): ClassificationResult[] {
    let embedding = baseModel.inferEmbeddingSync(file_or_image_tensor)
    let outputs = classifierModel.predict(embedding)
    embedding.dispose()
    let values = toOneTensor(outputs).dataSync()
    disposeTensor(outputs)
    return mapWithClassName(values)
  }

  function mapWithClassName(values: ArrayLike<number>): ClassificationResult[] {
    let result = new Array(class_names.length)
    for (let i = 0; i < class_names.length; i++) {
      result[i] = {
        label: class_names[i],
        score: values[i],
      }
    }
    return result
  }

  async function loadDatasetFromDirectoryAsync() {
    let x: tf.Tensor[] = []
    let y: tf.Tensor[] = []
    for (let class_idx = 0; class_idx < class_names.length; class_idx++) {
      let output = tf.oneHot(
        tf.tensor1d([class_idx], 'int32'),
        class_names.length,
      )
      let dir = join(datasetDir, class_names[class_idx])
      let filenames = await readdir(dir)
      for (let filename of filenames) {
        let file = join(dir, filename)
        let embedding = await baseModel.inferEmbeddingAsync(file)
        x.push(embedding)
        y.push(output)
      }
    }
    return {
      x: tf.concat(x),
      y: tf.concat(y),
    }
  }

  async function trainAsync(options?: tf.ModelFitArgs): Promise<tf.History> {
    if (!compiled) {
      compile()
    }
    let { x, y } = await loadDatasetFromDirectoryAsync()
    let history = await classifierModel.fit(x, y, {
      ...options,
      shuffle: true,
    })
    x.dispose()
    y.dispose()
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
    class_names,
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
    if (item.score > max.score) {
      max = item
    }
  }
  return max
}
