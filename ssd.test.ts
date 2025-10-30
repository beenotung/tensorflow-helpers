import * as tf from '@tensorflow/tfjs'
import { PreTrainedImageModels } from './image-model'
import { loadGraphModel } from './model'

// SSD params
let numAnchorsPerCell = 1
let numClasses = 1

// NMS params
let maxOutputSize = 1
let iouThreshold = 0.5
let scoreThreshold = 0.3

// spatial layers, total 1029 boxes
let spatial_layers = [
  // Scale 1: 28×28×40  (stride 8) -> 784 boxes
  {
    name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_5/project/BatchNorm/FusedBatchNormV3',
    layer: 5,
    shape: [1, 28, 28, 40],
  },
  // Scale 2: 14×14×112  (stride 16) -> 196 boxes
  {
    name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_11/project/BatchNorm/FusedBatchNormV3',
    layer: 11,
    shape: [1, 14, 14, 112],
  },
  // Scale 3: 7×7×160  (stride 32) -> 49 boxes
  {
    name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_14/project/BatchNorm/FusedBatchNormV3',
    layer: 14,
    shape: [1, 7, 7, 160],
  },
]
let spatial_node_names = spatial_layers.map(layer => layer.name)

async function test() {
  let spec = PreTrainedImageModels.mobilenet['mobilenet-v3-large-100']
  let dir = 'saved_model/mobilenet-v3-large-100'
  let backbone = await loadGraphModel({ dir })

  // image size: 224×224 (rgb)
  let input = tf.zeros([224, 224, 3]) as tf.Tensor3D

  debugger
  let result = await detectInCrop(backbone, input)
  debugger
}

async function detectInCrop(
  backbone: tf.GraphModel,
  croppedImage: tf.Tensor3D,
) {
  // 1. Resize to 224x224
  // 2. Normalize to [-1, 1]
  let input = tf.image
    .resizeBilinear(croppedImage, [224, 224])
    .div(127.5)
    .sub(1.0)
    .expandDims(0) as tf.Tensor4D

  // 3. extract spatial features
  let spatialFeatures = extractFeatures(backbone, input)

  // 4. SSD heads -> boxes + scores
  let ssd_result = ssdHead(spatialFeatures)
  let boxes = tf.concat(ssd_result.boxes)
  let scores = tf.concat(ssd_result.scores)

  // 5. Non-maximum suppression -> final boxes

  let results = tf.image.nonMaxSuppression(
    boxes,
    scores,
    maxOutputSize,
    iouThreshold,
    scoreThreshold,
  )

  return results
}

function extractFeatures(backbone: tf.GraphModel, input: tf.Tensor4D) {
  let features = backbone.execute(input, spatial_node_names) as tf.Tensor[]
  return features
}

function ssdHead(spatialFeatures: tf.Tensor[]) {
  let boxes = []
  let scores = []
  for (let spatialFeature of spatialFeatures) {
    // 1x1 conv -> 128 channels
    let proj = tf.layers
      .conv2d({
        filters: 128,
        kernelSize: 1,
        activation: 'linear',
      })
      .apply(spatialFeature)

    // Localization haed: 4 coords per anchor
    let loc = tf.layers
      .conv2d({
        filters: numAnchorsPerCell * 4,
        kernelSize: 3,
        padding: 'same',
      })
      .apply(proj) as tf.Tensor2D

    // Classification head
    let cls = tf.layers
      .conv2d({
        filters: numAnchorsPerCell * numClasses,
        kernelSize: 3,
        padding: 'same',
      })
      .apply(proj) as tf.Tensor1D

    boxes.push(loc)
    scores.push(cls)
  }
  return { boxes, scores }
}

/**
 * TODO apply post-training quantization after model is trained
 * # TensorFlow.js (post-training quantization)
```python
converter = tf.lite.TFLiteConverter.from_saved_model(...)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
tflite_model = converter.convert()
```
 * 
 */
function applyPostTrainingQuantization(model: tf.LayersModel) {
  throw new Error('todo')
}

test()
