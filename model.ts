import * as tf from '@tensorflow/tfjs-node'
import { existsSync, mkdirSync } from 'fs'
import { readFile, writeFile } from 'fs/promises'
import { join } from 'path'
import { cropAndResize, loadImageFileAsync, loadImageFileSync } from './image'

export type IOHandler = Exclude<Parameters<tf.GraphModel['save']>[0], string>
export type ModelArtifacts = Parameters<Required<IOHandler>['save']>[0]
export type SaveResult = ReturnType<Required<IOHandler>['save']>

export type Model = tf.GraphModel | tf.LayersModel

export async function saveModel(options: {
  model: Model
  dir: string
}): Promise<SaveResult> {
  let { dir, model } = options
  return await model.save({
    async save(modelArtifact: ModelArtifacts) {
      let weights = modelArtifact.weightData
      if (!weights) {
        throw new Error('missing weightData')
      }
      if (!Array.isArray(weights)) {
        weights = [weights]
      }
      mkdirSync(dir, { recursive: true })
      writeFile(join(dir, 'model.json'), JSON.stringify(modelArtifact))
      for (let i = 0; i < weights.length; i++) {
        writeFile(join(dir, `weight-${i}.bin`), Buffer.from(weights[i]))
      }
      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: 'JSON',
        },
      }
    },
  })
}

export async function loadGraphModel(options: {
  dir: string
}): Promise<tf.GraphModel> {
  let { dir } = options
  let model = await tf.loadGraphModel({
    async load() {
      let buffer = await readFile(join(dir, 'model.json'))
      let modelArtifact: ModelArtifacts = JSON.parse(buffer.toString())
      let weights = modelArtifact.weightData
      if (!weights) {
        throw new Error('missing weightData')
      }
      if (!Array.isArray(weights)) {
        weights = [weights]
      }
      for (let i = 0; i < weights.length; i++) {
        let buffer = await readFile(join(dir, `weight-${i}.bin`))
        weights[i] = new Uint8Array(buffer)
      }
      return modelArtifact
    },
  })
  return model
}

export async function cachedLoadGraphModel(options: {
  url: string
  dir: string
}): Promise<Model> {
  let { url: modelUrl, dir: modelDir } = options
  if (existsSync(modelDir)) {
    return await loadGraphModel(options)
  }
  let model = await tf.loadGraphModel(modelUrl, { fromTFHub: true })
  await saveModel({ model, dir: modelDir })
  return model
}

export type BaseModelSpec = {
  url: string
  width: number
  height: number
  channels: number
  features: number
}

export const baseModels = {
  mobilenet: {
    'mobilenet-v3-large-100': {
      url: 'https://www.kaggle.com/models/google/mobilenet-v3/TfJs/large-100-224-feature-vector/1' as const,
      version: 3 as const,
      alpha: 1.0 as const,
      width: 224 as const,
      height: 224 as const,
      channels: 3 as const,
      features: 1280 as const,
    },
  },
}

export type ImageBaseModel = Awaited<ReturnType<typeof loadImageBaseModel>>

export async function loadImageBaseModel(options: {
  spec: BaseModelSpec
  dir: string
}) {
  let { spec, dir } = options
  let { url, width, height, channels } = spec

  let model = await cachedLoadGraphModel({
    url,
    dir,
  })

  async function loadImageAsync(file: string): Promise<tf.Tensor4D> {
    let imageTensor = await loadImageFileAsync(file, {
      channels,
      expandAnimations: false,
    })
    imageTensor = cropAndResize({ imageTensor, width, height })
    return imageTensor
  }

  function loadImageSync(file: string): tf.Tensor4D {
    let imageTensor = loadImageFileSync(file, {
      channels,
      expandAnimations: false,
    })
    return cropAndResize({ imageTensor, width, height })
  }

  async function loadAnimatedImageAsync(file: string): Promise<tf.Tensor4D> {
    let imageTensor = await loadImageFileAsync(file, {
      channels,
      expandAnimations: true,
    })
    return cropAndResize({ imageTensor, width, height })
  }

  function loadAnimatedImageSync(file: string): tf.Tensor4D {
    let imageTensor = loadImageFileSync(file, {
      channels,
      expandAnimations: true,
    })
    imageTensor = cropAndResize({ imageTensor, width, height })
    return imageTensor
  }

  async function inferEmbeddingAsync(
    file_or_image_tensor: string | tf.Tensor,
  ): Promise<tf.Tensor> {
    let inputs: tf.Tensor =
      typeof file_or_image_tensor == 'string'
        ? await loadImageAsync(file_or_image_tensor)
        : file_or_image_tensor
    let outputs = (model as tf.GraphModel).predictAsync
      ? await (model as tf.GraphModel).predictAsync(inputs)
      : model.predict(inputs)
    if (typeof file_or_image_tensor == 'string') {
      inputs.dispose()
    }
    return Array.isArray(outputs) ? outputs[0] : (outputs as tf.Tensor)
  }

  function inferEmbeddingSync(
    file_or_image_tensor: string | tf.Tensor,
  ): tf.Tensor {
    let inputs: tf.Tensor =
      typeof file_or_image_tensor == 'string'
        ? loadImageSync(file_or_image_tensor)
        : file_or_image_tensor
    let outputs = model.predict(inputs)
    if (typeof file_or_image_tensor == 'string') {
      inputs.dispose()
    }
    return Array.isArray(outputs) ? outputs[0] : (outputs as tf.Tensor)
  }

  return {
    spec,
    model,
    loadImageAsync,
    loadImageSync,
    loadAnimatedImageAsync,
    loadAnimatedImageSync,
    inferEmbeddingAsync,
    inferEmbeddingSync,
  }
}
