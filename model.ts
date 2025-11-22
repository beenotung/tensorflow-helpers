import * as tf from '@tensorflow/tfjs'
import sharp, { Sharp } from 'sharp'
import { existsSync, mkdirSync } from 'fs'
import { mkdir, readFile, writeFile } from 'fs/promises'
import { basename, join } from 'path'
import {
  CropAndResizeAspectRatio,
  cropAndResizeImageSharp,
  cropAndResizeImageTensor,
  loadImageFile,
} from './image'
import { imageSharpToTensor } from './image-utils'
import { isContentHash } from './file'
import { toOneTensor } from './tensor'
import { ImageModelSpec } from './image-model'
import { getClassCount } from './classifier-utils'
import {
  filterSpatialNodesWithUniqueShapes,
  getSpatialNodes,
} from './spatial-utils'
import { ImageEmbeddingOptions } from './model-utils'
import {
  ModelArtifacts,
  ModelJSON,
  SaveResult,
} from '@tensorflow/tfjs-core/dist/io/types'
import {
  attachClassNames,
  checkClassNames,
  patchLoadedModelJSON,
  SavedModelJSON,
} from './internal'
export { ImageModelSpec, PreTrainedImageModels } from './image-model'

export type Model = tf.GraphModel | tf.LayersModel

export async function saveModel(options: {
  model: Model
  dir: string
  classNames?: string[]
}): Promise<SaveResult> {
  let { dir, model, classNames } = options
  return await model.save({
    async save(modelArtifact: ModelArtifacts) {
      await mkdir(dir, { recursive: true })
      let modelJSON = modelArtifact as ModelJSON
      if (
        modelArtifact.weightData &&
        modelArtifact.weightSpecs &&
        !('weightsManifest' in modelArtifact)
      ) {
        let { weightData, weightSpecs, ...rest } = modelArtifact
        modelJSON = rest as ModelJSON
        modelJSON.weightsManifest = [{ paths: [], weights: weightSpecs }]
        if (!Array.isArray(weightData)) {
          weightData = [weightData]
        }
        for (let i = 0; i < weightData.length; i++) {
          let filename = `group1-shard${i + 1}of${weightData.length}.bin`
          modelJSON.weightsManifest[0].paths.push(filename)
          let file = join(dir, filename)
          await writeFile(file, Buffer.from(weightData[i]))
        }
      }
      if (classNames) {
        modelJSON.userDefinedMetadata ||= {}
        modelJSON.userDefinedMetadata.classNames = classNames
      }
      await writeFile(join(dir, 'model.json'), JSON.stringify(modelJSON))

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
  classNames?: string[]
}) {
  let { dir, classNames } = options
  let buffer = await readFile(join(dir, 'model.json'))
  let modelArtifact: SavedModelJSON = JSON.parse(buffer.toString())
  let changed = patchLoadedModelJSON(modelArtifact)
  classNames = checkClassNames(modelArtifact, classNames)
  if (changed) {
    await writeFile(join(dir, 'model.json'), JSON.stringify(modelArtifact))
  }
  let model = await tf.loadGraphModel('file://' + join(dir, 'model.json'))
  return attachClassNames(model, classNames)
}

export async function loadLayersModel(options: {
  dir: string
  classNames?: string[]
}) {
  let { dir, classNames } = options
  let buffer = await readFile(join(dir, 'model.json'))
  let modelArtifact: SavedModelJSON = JSON.parse(buffer.toString())
  let changed = patchLoadedModelJSON(modelArtifact)
  classNames = checkClassNames(modelArtifact, classNames)
  if (changed) {
    await writeFile(join(dir, 'model.json'), JSON.stringify(modelArtifact))
  }
  let model = await tf.loadLayersModel('file://' + join(dir, 'model.json'))
  if (classNames) {
    let classCount = getClassCount(model.outputShape)
    if (classCount != classNames.length) {
      throw new Error(
        `number of classes mismatch, expected: ${classNames.length}, got: ${classCount}`,
      )
    }
  }
  return attachClassNames(model, classNames)
}

export async function cachedLoadGraphModel(options: {
  url: string
  dir: string
  classNames?: string[]
}) {
  let { url: modelUrl, dir: modelDir, classNames } = options
  if (existsSync(modelDir)) {
    return await loadGraphModel(options)
  }
  let model = await tf.loadGraphModel(modelUrl, { fromTFHub: true })
  await saveModel({ model, dir: modelDir, classNames })
  return attachClassNames(model, classNames)
}

export async function cachedLoadLayersModel(options: {
  url: string
  dir: string
  classNames?: string[]
}) {
  let { url: modelUrl, dir: modelDir, classNames } = options
  if (existsSync(modelDir)) {
    return await loadLayersModel(options)
  }
  let model = await tf.loadLayersModel(modelUrl, { fromTFHub: true })
  await saveModel({ model, dir: modelDir, classNames })
  return attachClassNames(model, classNames)
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

  function checkCache(
    file_or_filename: string,
    options?: ImageEmbeddingOptions,
  ): tf.Tensor | void {
    if (!fileEmbeddingCache || !isContentHash(file_or_filename)) return

    let filename = basename(file_or_filename)

    let embedding = fileEmbeddingCache.get(filename)
    if (embedding) {
      let shape = embedding.shape
      if (options?.squeeze && shape.length > 1 && shape[0] == 1) {
        let squeezed = tf.squeeze(embedding, [0])
        embedding.dispose()
        fileEmbeddingCache.set(filename, squeezed)
        return squeezed
      }
      return embedding
    }

    let values = typeof cache == 'object' ? cache.get(filename) : undefined
    if (!values) return

    embedding = options?.squeeze ? tf.tensor(values) : tf.tensor([values])
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
    options?: ImageEmbeddingOptions,
  ): Promise<tf.Tensor> {
    let embedding = checkCache(file, options)
    if (embedding) return embedding

    try {
      let image = sharp(file)
      embedding = await imageSharpToEmbedding(image)
    } catch (error) {
      throw new Error('failed to load image: ' + JSON.stringify(file), {
        cause: error,
      })
    }

    if (cache && isContentHash(file)) {
      saveCache(file, embedding)
    }
    return embedding
  }

  function imageTensorToEmbedding(
    imageTensor: tf.Tensor3D | tf.Tensor4D,
    options?: ImageEmbeddingOptions,
  ): tf.Tensor {
    return tf.tidy(() => {
      imageTensor = cropAndResizeImageTensor({
        imageTensor,
        width,
        height,
        aspectRatio,
      })
      let embedding = model.predict(imageTensor) as tf.Tensor
      if (options?.squeeze) {
        embedding = tf.squeeze(embedding, [0])
      }
      return embedding
    })
  }

  async function imageSharpToEmbedding(image: Sharp): Promise<tf.Tensor> {
    image = cropAndResizeImageSharp({ image, width, height, aspectRatio })
    let imageTensor = await imageSharpToTensor(image)
    let embedding = tf.tidy(() => {
      let outputs = model.predict(imageTensor)
      let embedding = toOneTensor(outputs)
      return embedding
    })
    imageTensor.dispose()
    return embedding
  }

  let spatialNodes = getSpatialNodes({ model, tf })
  let spatialNodesWithUniqueShapes =
    filterSpatialNodesWithUniqueShapes(spatialNodes)
  let lastSpatialNode = spatialNodesWithUniqueShapes.slice().pop()

  return {
    spec,
    model,
    fileEmbeddingCache,
    checkCache,
    loadImageCropped,
    imageFileToEmbedding,
    imageTensorToEmbedding,
    imageSharpToEmbedding,
    spatialNodes,
    spatialNodesWithUniqueShapes,
    lastSpatialNode,
  }
}
