import * as tf from '@tensorflow/tfjs-node'
import { existsSync, mkdirSync } from 'fs'
import { readFile, writeFile } from 'fs/promises'
import { basename, join } from 'path'
import {
  CropAndResizeAspectRatio,
  cropAndResizeImageTensor,
  loadImageFile,
} from './image'
import { isContentHash } from './file'
import { toOneTensor } from './tensor'
import { ImageModelSpec } from './image-model'
export { ImageModelSpec, PreTrainedImageModels } from './image-model'

export type IOHandler = Exclude<Parameters<tf.GraphModel['save']>[0], string>
export type ModelArtifacts = Parameters<Required<IOHandler>['save']>[0]
export type SaveResult = ReturnType<Required<IOHandler>['save']>

export type ModelArtifactsWithClassNames = ModelArtifacts & {
  classNames?: string[]
}

export type Model = tf.GraphModel | tf.LayersModel

export async function saveModel(options: {
  model: Model
  dir: string
  classNames?: string[]
}): Promise<SaveResult> {
  let { dir, model, classNames } = options
  return await model.save({
    async save(modelArtifact: ModelArtifactsWithClassNames) {
      if (classNames) {
        modelArtifact.classNames = classNames
      }
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
        weights[i] = await loadWeightData(join(dir, `weight-${i}.bin`))
      }
      return modelArtifact
    },
  })
  return model
}

export async function loadLayersModel(options: {
  dir: string
  classNames?: string[]
}): Promise<tf.LayersModel> {
  let { dir, classNames } = options
  let model = await tf.loadLayersModel({
    async load() {
      let buffer = await readFile(join(dir, 'model.json'))
      let modelArtifact: ModelArtifactsWithClassNames = JSON.parse(
        buffer.toString(),
      )
      if (
        classNames &&
        modelArtifact.classNames &&
        JSON.stringify(classNames) !== JSON.stringify(modelArtifact.classNames)
      ) {
        throw new Error(
          `classNames mismatch, expected: ${JSON.stringify(
            classNames,
          )}, actual: ${JSON.stringify(modelArtifact.classNames)}`,
        )
      }
      let weights = modelArtifact.weightData
      if (!weights) {
        throw new Error('missing weightData')
      }
      if (!Array.isArray(weights)) {
        modelArtifact.weightData = await loadWeightData(
          join(dir, `weight-0.bin`),
        )
        return modelArtifact
      }
      for (let i = 0; i < weights.length; i++) {
        weights[i] = await loadWeightData(join(dir, `weight-${i}.bin`))
      }
      return modelArtifact
    },
  })
  return model
}

async function loadWeightData(file: string) {
  let buffer = await readFile(file)
  return new Uint8Array(buffer)
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

export async function cachedLoadLayersModel(options: {
  url: string
  dir: string
}): Promise<Model> {
  let { url: modelUrl, dir: modelDir } = options
  if (existsSync(modelDir)) {
    return await loadLayersModel(options)
  }
  let model = await tf.loadLayersModel(modelUrl, { fromTFHub: true })
  await saveModel({ model, dir: modelDir })
  return model
}

export type ImageModel = Awaited<ReturnType<typeof loadImageModel>>

/**
 * @description cache image embedding keyed by filename.
 * The dirname is ignored.
 * The filename is expected to be content hash (w/wo extname)
 */
export type EmbeddingCache = {
  get(filename: string): number[] | null | undefined
  set(filename: string, values: number[]): void
}

export async function loadImageModel<Cache extends EmbeddingCache>(options: {
  spec: ImageModelSpec
  dir: string
  aspectRatio?: CropAndResizeAspectRatio
  cache?: Cache | boolean
}) {
  let { spec, dir, aspectRatio, cache } = options
  let { url, width, height, channels } = spec

  let model = await cachedLoadGraphModel({
    url,
    dir,
  })

  async function loadImageCropped(
    file: string,
    options?: {
      expandAnimations?: boolean
    },
  ): Promise<tf.Tensor3D | tf.Tensor4D> {
    let imageTensor = await loadImageFile(file, {
      channels,
      expandAnimations: options?.expandAnimations,
      crop: {
        width,
        height,
        aspectRatio,
      },
    })
    return imageTensor as tf.Tensor4D
  }

  let fileEmbeddingCache: Map<string, tf.Tensor<tf.Rank>> | null = cache
    ? new Map()
    : null

  function checkCache(file_or_filename: string): tf.Tensor | void {
    if (!fileEmbeddingCache || !isContentHash(file_or_filename)) return

    let filename = basename(file_or_filename)

    let embedding = fileEmbeddingCache.get(filename)
    if (embedding) return embedding

    let values = typeof cache == 'object' ? cache.get(filename) : undefined
    if (!values) return

    embedding = tf.tensor([values])
    fileEmbeddingCache.set(filename, embedding)

    return embedding
  }

  async function saveCache(file: string, embedding: tf.Tensor) {
    let filename = basename(file)

    fileEmbeddingCache!.set(filename, embedding)

    if (typeof cache == 'object') {
      let values = Array.from(await embedding.data())
      cache.set(filename, values)
    }
  }

  async function imageFileToEmbedding(
    file: string,
    options?: { expandAnimations?: boolean },
  ): Promise<tf.Tensor> {
    let embedding = checkCache(file)
    if (embedding) return embedding
    let content = await readFile(file)
    return tf.tidy(() => {
      let dtype = undefined
      let expandAnimations = options?.expandAnimations
      let imageTensor: tf.Tensor3D | tf.Tensor4D
      try {
        imageTensor = tf.node.decodeImage(
          content,
          channels,
          dtype,
          expandAnimations,
        )
      } catch (error) {
        throw new Error('failed to decode image: ' + JSON.stringify(file), {
          cause: error,
        })
      }
      let embedding = imageTensorToEmbedding(imageTensor)
      if (cache && isContentHash(file)) {
        saveCache(file, embedding)
      }
      return embedding
    })
  }

  function imageTensorToEmbedding(
    imageTensor: tf.Tensor3D | tf.Tensor4D,
  ): tf.Tensor {
    return tf.tidy(() => {
      imageTensor = cropAndResizeImageTensor({
        imageTensor,
        width,
        height,
        aspectRatio,
      })
      let outputs = model.predict(imageTensor)
      let embedding = toOneTensor(outputs)
      return embedding
    })
  }

  return {
    spec,
    model,
    fileEmbeddingCache,
    checkCache,
    loadImageCropped,
    imageFileToEmbedding,
    imageTensorToEmbedding,
  }
}
