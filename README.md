# tensorflow-helpers

Helper functions to use tensorflow in nodejs for transfer learning, image classification, and more.

[![npm Package Version](https://img.shields.io/npm/v/tensorflow-helpers)](https://www.npmjs.com/package/tensorflow-helpers)

## Features

- Support transfer learning and continuous learning
- Custom image classifier using embedding features from pre-trained image model
- Correctly save/load model on filesystem[1]
- Load image file into tensor with resize and crop
- List varies pre-trained models (url, image dimension, embedding size)
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
  // classNames: ['anime', 'real'], // auto scan from datasetDir
})

// auto load training dataset
let history = await classifier.trainAsync({
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

```typescript
import * as tfjs from '@tensorflow/tfjs-node'

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
  has(filename: string): boolean
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
<summary>Image Helper Functions</summary>

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
  trainAsync: (options?: tf.ModelFitArgs) => Promise<tf.History>
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
