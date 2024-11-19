import * as tf from '@tensorflow/tfjs'
import { ModelArtifacts } from '@tensorflow/tfjs-core/dist/io/types'
import {
  CropAndResizeAspectRatio,
  ImageTensor,
  cropAndResizeImageTensor,
} from '../image-utils'
import { ImageModelSpec } from '../image-model'
import { toOneTensor } from '../tensor'

export type ModelArtifactsWithClassNames = ModelArtifacts & {
  classNames?: string[]
}

async function readFile(url: string) {
  let res = await fetch(url)
  let buffer = await res.arrayBuffer()
  return buffer
}

async function loadWeightData(file: string) {
  let buffer = await readFile(file)
  return new Uint8Array(buffer)
}

async function readJSON(url: string) {
  let res = await fetch(url)
  if (res.status == 404) {
    throw new Error('json file not found: ' + url)
  }
  let json = await res.json()
  return json
}

function removeModelUrlPrefix(url: string) {
  if (url.endsWith('/model.json')) {
    url = url.slice(0, url.length - '/model.json'.length)
  }
  if (url.endsWith('/')) {
    url = url.slice(0, url.length - 1)
  }
  return url
}

async function getLastModified(url: string) {
  url = removeModelUrlPrefix(url)
  url += '/model.json'
  let res = await fetch(url, { method: 'HEAD' })
  if (res.status == 404) {
    throw new Error('file not found: ' + url)
  }
  let text = res.headers.get('Last-Modified')
  return text ? new Date(text).getTime() : Date.now()
}

/**
 * @example `loadGraphModel({ url: 'saved_model/mobilenet-v3-large-100' })`
 */
export async function loadGraphModel(options: {
  url: string
}): Promise<tf.GraphModel> {
  let url = removeModelUrlPrefix(options.url)
  let model = await tf.loadGraphModel({
    async load() {
      let modelArtifact: ModelArtifacts = await readJSON(url + '/model.json')
      let weights = modelArtifact.weightData
      if (!weights) {
        throw new Error('missing weightData')
      }
      if (!Array.isArray(weights)) {
        weights = [weights]
      }
      for (let i = 0; i < weights.length; i++) {
        weights[i] = await loadWeightData(url + `/weight-${i}.bin`)
      }
      return modelArtifact
    },
  })
  return model
}

async function cachedLoadModel<
  Model extends { save(url: string): Promise<unknown> },
>(args: {
  options: {
    url: string
    cacheUrl: string
    checkForUpdates?: boolean
  }
  loadRemoteModel(): Promise<Model>
  loadLocalModel(): Promise<Model>
}) {
  let { options, loadRemoteModel, loadLocalModel } = args
  let { url: modelUrl, cacheUrl } = options

  let localLastModified = +localStorage.getItem(cacheUrl)!
  let checkTime = Date.now()
  let remoteLastModified = options.checkForUpdates
    ? await getLastModified(modelUrl).catch(error => {
        if (localLastModified) {
          // skip checking if offline and already cached
          return localLastModified
        }
        // throw error if offline without pre-cached copy
        throw error
      })
    : 0

  if (
    localLastModified &&
    (!options.checkForUpdates || localLastModified == remoteLastModified)
  ) {
    try {
      let model = await loadLocalModel()
      return model
    } catch (error) {
      if (!String(error).includes('Cannot find model with path')) {
        throw error
      }
    }
  }

  let model = await loadRemoteModel()

  await model.save(cacheUrl)
  localStorage.setItem(cacheUrl, (remoteLastModified || checkTime).toString())

  return model
}

/**
 * @example `loadGraphModel({ url: 'saved_model/emotion-classifier' })`
 */
export async function loadLayersModel(options: {
  url: string
  classNames?: string[]
}) {
  let url = removeModelUrlPrefix(options.url)
  let classNames = options.classNames
  let model = await tf.loadLayersModel({
    async load() {
      let modelArtifact: ModelArtifactsWithClassNames = await readJSON(
        url + '/model.json',
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
        modelArtifact.weightData = await loadWeightData(url + `/weight-0.bin`)
        return modelArtifact
      }
      for (let i = 0; i < weights.length; i++) {
        weights[i] = await loadWeightData(url + `/weight-${i}.bin`)
      }
      return modelArtifact
    },
  })
  return model
}

/**
 * @example ```
 * cachedLoadGraphModel({
 *   url: 'saved_model/mobilenet-v3-large-100',
 *   cacheUrl: 'indexeddb://mobilenet-v3-large-100',
 * })
 * ```
 */
export async function cachedLoadGraphModel(options: {
  url: string
  cacheUrl: string
  checkForUpdates?: boolean
}) {
  return cachedLoadModel({
    options,
    loadRemoteModel: () => loadGraphModel(options),
    loadLocalModel: () => tf.loadGraphModel(options.cacheUrl),
  })
}

/**
 * @example ```
 * cachedLoadLayersModel({
 *   url: 'saved_model/emotion-classifier',
 *   cacheUrl: 'indexeddb://emotion-classifier',
 * })
 * ```
 */
export async function cachedLoadLayersModel(options: {
  url: string
  cacheUrl: string
  checkForUpdates?: boolean
}) {
  return cachedLoadModel({
    options,
    loadRemoteModel: () => loadLayersModel(options),
    loadLocalModel: () => tf.loadLayersModel(options.cacheUrl),
  })
}

export type ImageModel = Awaited<ReturnType<typeof loadImageModel>>

type ModelWithSignature = {
  signature: {
    inputs: SignatureTensorMap
    outputs: SignatureTensorMap
  }
}
type SignatureTensorMap = {
  [name: string]: {
    tensorShape: { dim: { size: string }[] }
  }
}

function getInt(str: string) {
  let int = +str
  if (int && Number.isInteger(int)) {
    return int
  }
  throw new TypeError(`expect int value, got: ${JSON.stringify(str)}`)
}

function getModelSpec(url: string, model: tf.GraphModel) {
  let { signature } = model as any as ModelWithSignature

  let inputs = Object.values(signature.inputs)[0]
  let outputs = Object.values(signature.outputs)[0]

  let height = getInt(inputs.tensorShape.dim[1].size)
  let width = getInt(inputs.tensorShape.dim[2].size)
  let channels = getInt(inputs.tensorShape.dim[3].size)
  let features = getInt(outputs.tensorShape.dim[1].size)

  let spec: ImageModelSpec = {
    url,
    width,
    height,
    channels,
    features,
  }

  return spec
}

function basename(url: string) {
  if (url.startsWith('data:')) {
    return ''
  }
  return url.split('#')[0].split('?')[0].split('/').pop()!
}

function isContentHash(url: string): boolean {
  let filename = basename(url)
  let ext = filename.split('.').pop()!
  let name = ext.length == 0 ? filename : filename.slice(0, -(ext.length + 1))
  return name.length * 4 == 256 && isHexOnly(name)
}

function isHexOnly(str: string) {
  for (let char of str) {
    if (str >= '0' && str <= '9') continue
    if (str >= 'A' && str <= 'F') continue
    if (str >= 'a' && str <= 'f') continue
    return false
  }
  return true
}

/**
 * @description cache image embedding keyed by filename.
 * The dirname is ignored.
 * The filename is expected to be content hash (w/wo extname)
 */
export type EmbeddingCache = {
  get(url: string): number[] | null | undefined
  set(url: string, values: number[]): void
}

export async function loadImageModel<Cache extends EmbeddingCache>(options: {
  url: string
  cacheUrl?: string
  checkForUpdates?: boolean
  aspectRatio?: CropAndResizeAspectRatio
  cache?: Cache | boolean
}) {
  let { aspectRatio, cache } = options

  let model = options.cacheUrl
    ? await cachedLoadGraphModel({
        url: options.url,
        cacheUrl: options.cacheUrl,
        checkForUpdates: options.checkForUpdates,
      })
    : await tf.loadGraphModel(options.url)
  let spec = getModelSpec(options.url, model)
  let { width, height, channels } = spec

  async function loadImageCropped(url: string) {
    let image = new Image()
    let p = new Promise((resolve, reject) => {
      ;(image.onload = resolve),
        (image.onerror = error =>
          reject(
            new Error('failed to load image: ' + JSON.stringify(url), {
              cause: error,
            }),
          ))
    })
    image.src = url
    await p
    let imageTensor = tf.browser.fromPixels(image, channels)
    return cropAndResizeImageTensor({
      imageTensor,
      width,
      height,
      aspectRatio,
    })
  }

  let fileEmbeddingCache: Map<string, tf.Tensor<tf.Rank>> | null = cache
    ? new Map()
    : null

  function checkCache(url: string): tf.Tensor | void {
    if (!fileEmbeddingCache || !isContentHash(url)) return

    let filename = basename(url)

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

  async function imageUrlToEmbedding(url: string): Promise<tf.Tensor> {
    let embedding = checkCache(url)
    if (embedding) return embedding

    let imageTensor = await loadImageCropped(url)

    embedding = imageTensorToEmbedding(imageTensor)
    imageTensor.dispose()

    if (cache && isContentHash(url)) {
      saveCache(url, embedding)
    }

    return embedding
  }

  async function imageFileToEmbedding(file: File): Promise<tf.Tensor> {
    let filename = file.name
    let embedding = checkCache(filename)
    if (embedding) return embedding

    let url = await new Promise<string>((resolve, reject) => {
      let reader = new FileReader()
      reader.onload = () => resolve(reader.result as string)
      reader.onerror = reject
      reader.readAsDataURL(file)
    })

    let imageTensor = await loadImageCropped(url)

    embedding = imageTensorToEmbedding(imageTensor)
    imageTensor.dispose()

    if (cache && isContentHash(filename)) {
      saveCache(filename, embedding)
    }

    return embedding
  }

  function imageTensorToEmbedding(imageTensor: ImageTensor): tf.Tensor {
    return tf.tidy(() => {
      let inputTensor = cropAndResizeImageTensor({
        imageTensor,
        width,
        height,
        aspectRatio,
      })
      let outputs = model.predict(inputTensor)
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
    imageUrlToEmbedding,
    imageFileToEmbedding,
    imageTensorToEmbedding,
  }
}
