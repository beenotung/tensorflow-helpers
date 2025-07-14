# tensorflow-helpers

Helper functions to use tensorflow in nodejs/browser for transfer learning, image classification, and more.

[![npm Package Version](https://img.shields.io/npm/v/tensorflow-helpers)](https://www.npmjs.com/package/tensorflow-helpers)

## Features

- Support transfer learning and continuous learning
- Custom image classifier using embedding features from pre-trained image model
- Correctly save/load model on filesystem[1]
- Load image file into tensor with resize and crop
- List varies pre-trained models (url, image dimension, embedding size)
- Support both nodejs and browser environment
- Support caching model and image embedding
- Typescript support
- Works with plain Javascript, Typescript is not mandatory

[1]: The built-in `tf.loadGraphModel()` cannot load the model saved by `model.save()`

## Installation

```bash
npm install tensorflow-helpers
```

You can also install `tensorflow-helpers` with [pnpm](https://pnpm.io/), [yarn](https://yarnpkg.com/), or [slnpm](https://github.com/beenotung/slnpm)

## Usage Example

See [model.test.ts](./model.test.ts) and [classifier.test.ts](./classifier.test.ts) for complete examples.

**Usage from browser**:

```typescript
import {
  loadImageClassifierModel,
  loadImageModel,
  toOneTensor,
} from 'tensorflow-helpers/browser'

declare var fileInput: HTMLInputElement

async function main() {
  let baseModel = await loadImageModel({
    url: 'saved_model/mobilenet-v3-large-100',
    cacheUrl: 'indexeddb://mobilenet-v3-large-100',
    checkForUpdates: false,
  })

  let classifier = await loadImageClassifierModel({
    baseModel,
    classNames: ['anime', 'real', 'others'],
    modelUrl: 'saved_model/emotion-classifier',
    cacheUrl: 'indexeddb://emotion-classifier',
  })

  fileInput.onchange = async () => {
    let file = fileInput.files?.[0]
    if (!file) return

    /* load from file directly */
    let result1 = await classifier.classifyImageFile(file)
    // result1 is Array<{ label: string, confidence: number }>

    /* load from embedding tensor */
    let embeddingTensor = await baseModel.imageFileToEmbedding(file)
    let embeddingFeatures = toOneTensor(embeddingTensor).dataSync()
    // embeddingFeatures is Float32Array

    let result1 = await classifier.classifyImageEmbedding(embeddingTensor)
  }
}
main().catch(e => console.error(e))
```

**Usage from nodejs**:

```typescript
import {
  loadImageModel,
  loadImageClassifierModel,
  PreTrainedImageModels,
} from 'tensorflow-helpers'

// auto cache locally
let baseModel = await loadImageModel({
  spec: PreTrainedImageModels.mobilenet['mobilenet-v3-large-100'],
  dir: 'saved_model/base_model',
})
console.log('embedding features:', baseModel.spec.features)
// [print] embedding features: 1280

// restore or create new model
let classifier = await loadImageClassifierModel({
  baseModel,
  modelDir: 'saved_model/classifier_model',
  hiddenLayers: [128],
  datasetDir: 'dataset',
  // classNames: ['anime', 'real', 'others'], // auto scan from datasetDir
})

// auto load training dataset
let history = await classifier.train({
  epochs: 5,
  batchSize: 32,
})

// persist the parameters across restart
await classifier.save()

// auto load image from filesystem, resize and crop
let classes = await classifier.classifyAsync('image.jpg')
let topClass = topClassificationResult(classes)

console.log('result:', topClass)
// [print] result: { label: 'anime', confidence: 0.7991582155227661 }
```

## Typescript Signature

Details see the type hints from IDE.

<details>
<summary>Shortcut to tensorflow</summary>

exported as `'tensorflow-helpers'`:

```typescript
import * as tfjs from '@tensorflow/tfjs-node'

export let tensorflow: typeof tfjs
export let tf: typeof tfjs
```

exported as `'tensorflow-helpers/browser'`:

```typescript
import * as tfjs from '@tensorflow/tfjs'

export let tensorflow: typeof tfjs
export let tf: typeof tfjs
```

</details>

<details>
<summary>Pre-trained model constants</summary>

```typescript
export const PreTrainedImageModels: {
  mobilenet: {
    'mobilenet-v3-large-100': {
      url: 'https://www.kaggle.com/models/google/mobilenet-v3/TfJs/large-100-224-feature-vector/1'
      width: 224
      height: 224
      channels: 3
      features: 1280
    }
    // more models omitted ...
  }
}
```

</details>

<details>
<summary>Model helper functions</summary>

```typescript
export type Model = tf.GraphModel | tf.LayersModel

export function saveModel(options: {
  model: Model
  dir: string
}): Promise<SaveResult>

export function loadGraphModel(options: { dir: string }): Promise<tf.GraphModel>

export function loadLayersModel(options: {
  dir: string
}): Promise<tf.LayersModel>

export function cachedLoadGraphModel(options: {
  url: string
  dir: string
}): Promise<Model>

export function cachedLoadLayersModel(options: {
  url: string
  dir: string
}): Promise<Model>

export function loadImageModel(options: {
  spec: ImageModelSpec
  dir: string
  aspectRatio?: CropAndResizeAspectRatio
  cache?: EmbeddingCache | boolean
}): Promise<ImageModel>

export type EmbeddingCache = {
  get(filename: string): number[] | null | undefined
  set(filename: string, values: number[]): void
}

export type ImageModelSpec = {
  url: string
  width: number
  height: number
  channels: number
  features: number
}

export type ImageModel = {
  spec: ImageModelSpec
  model: Model

  fileEmbeddingCache: Map<string, tf.Tensor> | null
  checkCache(file_or_filename: string): tf.Tensor | void

  loadImageCropped(
    file: string,
    options?: {
      expandAnimations?: boolean
    },
  ): Promise<tf.Tensor3D | tf.Tensor4D>

  imageFileToEmbedding(
    file: string,
    options?: {
      expandAnimations?: boolean
    },
  ): Promise<tf.Tensor>

  imageTensorToEmbedding(imageTensor: tf.Tensor3D | tf.Tensor4D): tf.Tensor
}
```

</details>

<details>
<summary>Image helper functions and types</summary>

```typescript
export function loadImageFile(
  file: string,
  options?: {
    channels?: number
    dtype?: string
    expandAnimations?: boolean
    crop?: {
      width: number
      height: number
      aspectRatio?: CropAndResizeAspectRatio
    }
  },
): Promise<tf.Tensor3D | tf.Tensor4D>

export type ImageTensor = tf.Tensor3D | tf.Tensor4D

export function getImageTensorShape(imageTensor: tf.Tensor3D | tf.Tensor4D): {
  width: number
  height: number
}

export type Box = [top: number, left: number, bottom: number, right: number]

/**
 * @description calculate center-crop box
 * @returns [top,left,bottom,right], values range: 0..1
 */
export function calcCropBox(options: {
  sourceShape: { width: number; height: number }
  targetShape: { width: number; height: number }
}): Box

/**
 * @description default is 'rescale'
 *
 * 'rescale' -> scratch/transform to target shape;
 *
 * 'center-crop' -> crop the edges, maintain aspect ratio at center
 */
export type CropAndResizeAspectRatio = 'rescale' | 'center-crop'

export function cropAndResizeImageTensor(options: {
  imageTensor: tf.Tensor3D | tf.Tensor4D
  width: number
  height: number
  aspectRatio?: CropAndResizeAspectRatio
}): tf.Tensor4D

export function cropAndResizeImageFile(options: {
  srcFile: string
  destFile: string
  width: number
  height: number
  aspectRatio?: CropAndResizeAspectRatio
}): Promise<void>
```

</details>

<details>
<summary>Tensor helper functions</summary>

```typescript
export function disposeTensor(tensor: tf.Tensor | tf.Tensor[]): void

export function toOneTensor(
  tensor: tf.Tensor | tf.Tensor[] | tf.NamedTensorMap,
): tf.Tensor

export function toTensor4D(tensor: tf.Tensor3D | tf.Tensor4D): tf.Tensor4D

export function toTensor3D(tensor: tf.Tensor3D | tf.Tensor4D): tf.Tensor3D
```

</details>

<details>
<summary>Classifier helper functions</summary>

```typescript
export type ClassifierModelSpec = {
  embeddingFeatures: number
  hiddenLayers?: number[]
  classes: number
}

export function createImageClassifier(spec: ClassifierModelSpec): tf.Sequential

export type ClassificationResult = {
  label: string
  /** @description between 0 to 1 */
  confidence: number
}

export type ClassifierModel = {
  baseModel: {
    spec: ImageModelSpec
    model: Model
    loadImageAsync: (file: string) => Promise<tf.Tensor4D>
    loadImageSync: (file: string) => tf.Tensor4D
    loadAnimatedImageAsync: (file: string) => Promise<tf.Tensor4D>
    loadAnimatedImageSync: (file: string) => tf.Tensor4D
    inferEmbeddingAsync: (
      file_or_image_tensor: string | tf.Tensor,
    ) => Promise<tf.Tensor>
    inferEmbeddingSync: (file_or_image_tensor: string | tf.Tensor) => tf.Tensor
  }
  classifierModel: tf.LayersModel | tf.Sequential
  classNames: string[]
  classifyAsync: (
    file_or_image_tensor: string | tf.Tensor,
  ) => Promise<ClassificationResult[]>
  classifySync: (
    file_or_image_tensor: string | tf.Tensor,
  ) => ClassificationResult[]
  loadDatasetFromDirectoryAsync: () => Promise<{
    x: tf.Tensor<tf.Rank>
    y: tf.Tensor<tf.Rank>
  }>
  compile: () => void
  train: (options?: tf.ModelFitArgs) => Promise<tf.History>
  save: (dir?: string) => Promise<SaveResult>
}

export function loadImageClassifierModel(options: {
  baseModel: ImageModel
  hiddenLayers?: number[]
  modelDir: string
  datasetDir: string
  classNames?: string[]
}): Promise<ClassifierModel>

export function topClassifyResult(
  items: ClassificationResult[],
): ClassificationResult

/**
 * @description the values is returned as is.
 * It should has be applied softmax already
 * */
export function mapWithClassName(
  classNames: string[],
  values: ArrayLike<number>,
  options?: {
    sort?: boolean
  },
): ClassificationResult[]
```

</details>

<details>
<summary>Model helper functions</summary>

```typescript
/**
 * A factor to give larger hidden layer size for complex tasks:
 * - 1 for easy tasks
 * - 2-3 for medium difficulty tasks
 * - 4-5 for complex tasks
 *
 * Remark: giving too high difficulty may result in over-fitting.
 */
export type Difficulty = number

/** Formula `hiddenSize = difficulty * sqrt(inputSize * outputSize)` */
export function calcHiddenLayerSize(options: {
  inputSize: number
  outputSize: number
  difficulty?: Difficulty
})

/** Inject one or more hidden layers that's having large gap between input size and output size. */
export function injectHiddenLayers(options: {
  layers: number[]
  difficulty?: Difficulty
  numberOfHiddenLayers?: number
})
```

</details>

<details>
<summary>File helper functions</summary>

```typescript
/**
 * @description
 * - rename filename to content hash + extname;
 * - return list of (renamed) filenames
 */
export async function scanDir(dir: string): Promise<string[]>

export function isContentHash(file_or_filename: string): boolean

export async function saveFile(args: {
  dir: string
  content: Buffer
  mimeType: string
}): Promise<void>

export function hashContent(
  content: Buffer,
  encoding: BufferEncoding = 'hex',
): string

/** @returns new filename with content hash and extname */
export async function renameFileByContentHash(file: string): Promise<string>
```

</details>

<details>
<summary>(Browser version) model functions and types</summary>

````typescript
/**
 * @example `loadGraphModel({ url: 'saved_model/mobilenet-v3-large-100' })`
 */
export function loadGraphModel(options: { url: string }): Promise<tf.GraphModel>

/**
 * @example `loadGraphModel({ url: 'saved_model/emotion-classifier' })`
 */
export function loadLayersModel(options: {
  url: string
}): Promise<tf.LayersModel>

/**
 * @example ```
 * cachedLoadGraphModel({
 *   url: 'saved_model/mobilenet-v3-large-100',
 *   cacheUrl: 'indexeddb://mobilenet-v3-large-100',
 * })
 * ```
 */
export function cachedLoadGraphModel(options: {
  url: string
  cacheUrl: string
  checkForUpdates?: boolean
}): Promise<tf.GraphModel<string | tf.io.IOHandler>>

/**
 * @example ```
 * cachedLoadLayersModel({
 *   url: 'saved_model/emotion-classifier',
 *   cacheUrl: 'indexeddb://emotion-classifier',
 * })
 * ```
 */
export function cachedLoadLayersModel(options: {
  url: string
  cacheUrl: string
  checkForUpdates?: boolean
}): Promise<tf.LayersModel>
````

</details>

<details>
<summary>(Browser version) image model functions and types</summary>

```typescript
export type ImageModel = {
  spec: ImageModelSpec
  model: tf.GraphModel<string | tf.io.IOHandler>
  fileEmbeddingCache: Map<string, tf.Tensor<tf.Rank>> | null
  checkCache: (url: string) => tf.Tensor | void
  loadImageCropped: (url: string) => Promise<tf.Tensor4D & tf.Tensor<tf.Rank>>
  imageUrlToEmbedding: (url: string) => Promise<tf.Tensor>
  imageFileToEmbedding: (file: File) => Promise<tf.Tensor>
  imageTensorToEmbedding: (imageTensor: ImageTensor) => tf.Tensor
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

export function loadImageModel<Cache extends EmbeddingCache>(options: {
  url: string
  cacheUrl?: string
  checkForUpdates?: boolean
  aspectRatio?: CropAndResizeAspectRatio
  cache?: Cache | boolean
}): Promise<ImageModel>
```

</details>

<details>
<summary>(Browser version) classifier functions and types</summary>

```typescript
export type ClassifierModel = {
  baseModel: ImageModel
  classifierModel: tf.LayersModel | tf.Sequential
  classNames: string[]
  classifyImageUrl(url: string): Promise<ClassificationResult[]>
  classifyImageFile(file: File): Promise<ClassificationResult[]>
  classifyImageTensor(
    imageTensor: tf.Tensor3D | tf.Tensor4D,
  ): Promise<ClassificationResult[]>
  classifyImage(
    image: Parameters<typeof tf.browser.fromPixels>[0],
  ): Promise<ClassificationResult[]>
  classifyImageEmbedding(embedding: tf.Tensor): Promise<ClassificationResult[]>
  compile(): void
  train(
    options: tf.ModelFitArgs & {
      x: tf.Tensor<tf.Rank>
      y: tf.Tensor<tf.Rank>
      /** @description to calculate classWeight */
      classCounts?: number[]
    },
  ): Promise<tf.History>
}

export function loadImageClassifierModel(options: {
  baseModel: ImageModel
  hiddenLayers?: number[]
  modelUrl?: string
  cacheUrl?: string
  checkForUpdates?: boolean
  classNames: string[]
}): Promise<ClassifierModel>
```

</details>
