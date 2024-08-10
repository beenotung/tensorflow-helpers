# tensorflow-helpers

Helper functions to use tensorflow in nodejs

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
  hidden_layers: [128],
  datasetDir: 'dataset',
  // class_names: ['anime', 'real'], // auto scan from datasetDir
})

// auto load training dataset
let history = await classifier.trainAsync({
  epochs: 5,
  batchSize: 32,
})

// to persist the parameters across restart
await classifier.save()

// auto load from filesystem, resize and crop
let classes = await classifier.classifyAsync('image.jpg')
let topClass = topClassificationResult(classes)

console.log('result:', topClass)
// [print] result: { label: 'anime', score: 0.7991582155227661 }
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

export function loadGraphModel(options: {
  dir: string
}): Promise<tf.GraphModel>

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
  loadImageAsync: (file: string) => Promise<tf.Tensor4D>
  loadImageSync: (file: string) => tf.Tensor4D
  loadAnimatedImageAsync: (file: string) => Promise<tf.Tensor4D>
  loadAnimatedImageSync: (file: string) => tf.Tensor4D
  inferEmbeddingAsync: (
    file_or_image_tensor: string | tf.Tensor,
  ) => Promise<tf.Tensor>
  inferEmbeddingSync: (file_or_image_tensor: string | tf.Tensor) => tf.Tensor
}

export function loadImageModel(options: {
  spec: ImageModelSpec
  dir: string
}): Promise<ImageModel>
```

</details>

<details>
<summary>Image Helper Functions</summary>

```typescript
export function loadImageFileAsync(
  file: string,
  options?: {
    channels?: number
    dtype?: string
    expandAnimations?: boolean
  },
): Promise<tf.Tensor3D | tf.Tensor4D>

export function loadImageFileSync(
  file: string,
  options?: {
    channels?: number
    dtype?: string
    expandAnimations?: boolean
  },
): tf.Tensor3D | tf.Tensor4D

export function getImageTensorShape(imageTensor: tf.Tensor3D | tf.Tensor4D): {
  width: number
  height: number
}

export function cropAndResize(options: {
  imageTensor: tf.Tensor3D | tf.Tensor4D
  width: number
  height: number
}): tf.Tensor4D
```

</details>

<details>
<summary>Tensor helper functions</summary>

```typescript
export function disposeTensor(tensor: tf.Tensor | tf.Tensor[]): void

export function toOneTensor(tensor: tf.Tensor | tf.Tensor[]): tf.Tensor<tf.Rank>
```

</details>
