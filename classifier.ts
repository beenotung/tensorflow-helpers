import { ImageModel, loadLayersModel, saveModel } from './model'
import * as tf from '@tensorflow/tfjs-node'
import { readdirSync, existsSync } from 'fs'
import { readdir } from 'fs/promises'
import { join } from 'path'

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

export async function loadImageClassifierModel(options: {
  baseModel: ImageModel
  hidden_layers?: number[]
  classifierModelDir: string
  datasetDir: string
  class_names?: string[]
}) {
  let { baseModel, datasetDir, classifierModelDir } = options

  let class_names = options.class_names || readdirSync(datasetDir)

  let classifierModel = existsSync(classifierModelDir)
    ? await loadLayersModel({ dir: classifierModelDir })
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
  ): Promise<tf.Tensor> {
    let embedding = await baseModel.inferEmbeddingAsync(file_or_image_tensor)
    return classifyAndDisposeEmbeddings(embedding)
  }

  function classifySync(file_or_image_tensor: string | tf.Tensor): tf.Tensor {
    let embedding = baseModel.inferEmbeddingSync(file_or_image_tensor)
    return classifyAndDisposeEmbeddings(embedding)
  }

  function classifyAndDisposeEmbeddings(embedding: tf.Tensor) {
    let outputs = classifierModel.predict(embedding)
    embedding.dispose()
    return Array.isArray(outputs) ? outputs[0] : (outputs as tf.Tensor)
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

  async function save(dir = classifierModelDir) {
    return await saveModel({
      model: classifierModel,
      dir,
    })
  }

  return {
    baseModel,
    classifierModel,
    classifyAsync,
    classifySync,
    loadDatasetFromDirectoryAsync,
    trainAsync,
    save,
  }
}
