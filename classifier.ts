import { ImageBaseModel } from './model'
import * as tf from '@tensorflow/tfjs-node'
import { readdir } from 'fs/promises'
import { join } from 'path'

export async function createImageClassifier(options: {
  baseModel: ImageBaseModel
  hidden_layers?: number[]
  datasetDir: string
  class_names?: string[]
}) {
  let { baseModel, hidden_layers, datasetDir } = options

  let class_names = options.class_names || (await readdir(datasetDir))

  let classifierModel = tf.sequential()
  classifierModel.add(
    tf.layers.inputLayer({ inputShape: [baseModel.spec.features] }),
  )
  classifierModel.add(tf.layers.globalAveragePooling1d())
  classifierModel.add(tf.layers.dropout({ rate: 0.2 }))
  if (hidden_layers) {
    for (let i = 0; i < hidden_layers.length; i++) {
      classifierModel.add(
        tf.layers.dense({ units: hidden_layers[i], activation: 'gelu' }),
      )
    }
  }
  classifierModel.add(
    tf.layers.dense({ units: class_names.length, activation: 'softmax' }),
  )

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

  async function trainAsync(options: tf.ModelFitArgs): Promise<tf.History> {
    let x: tf.Tensor[] = []
    let y: tf.Tensor[] = []
    for (let class_idx = 0; class_idx < class_names.length; class_idx++) {
      let output = tf.oneHot(
        tf.tensor1d([class_idx], 'int32'),
        class_names.length,
      )
      tf.oneHot(tf.tensor1d([0, 1], 'float32'), 3)
      let dir = join(datasetDir, class_names[class_idx])
      let filenames = await readdir(dir)
      for (let filename of filenames) {
        let file = join(dir, filename)
        let embedding = await baseModel.inferEmbeddingAsync(file)
        x.push(embedding)
        y.push(output)
      }
    }
    return await classifierModel.fit(x, y, {
      ...options,
      shuffle: true,
    })
  }

  return { baseModel, classifierModel, classifyAsync, classifySync, trainAsync }
}
