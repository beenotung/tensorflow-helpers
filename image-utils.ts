import './polyfill'
import * as tf from '@tensorflow/tfjs-core'
import { toTensor4D } from './tensor'

export type ImageTensor = tf.Tensor3D | tf.Tensor4D

export function getImageTensorShape(imageTensor: ImageTensor) {
  return imageTensor.shape.length === 3
    ? {
        width: imageTensor.shape[1],
        height: imageTensor.shape[0],
      }
    : {
        width: imageTensor.shape[2],
        height: imageTensor.shape[1],
      }
}

export type Box = [top: number, left: number, bottom: number, right: number]

/**
 * @description calculate center-crop box
 * @returns [top,left,bottom,right], values range: 0..1
 */
export function calcCropBox(options: {
  sourceShape: { width: number; height: number }
  targetShape: { width: number; height: number }
}): Box {
  let { sourceShape, targetShape } = options

  let sourceRatio = sourceShape.width / sourceShape.height
  let targetRatio = targetShape.width / targetShape.height

  let top = 0
  let left = 0
  let bottom = 1
  let right = 1

  // source too wide -> crop left/right
  if (sourceRatio > targetRatio) {
    let newWidth = targetRatio * sourceShape.height
    left = (sourceShape.width - newWidth) / 2 / sourceShape.width
    right = 1 - left
  }

  // source too tall -> crop top/bottom
  else if (sourceRatio < targetRatio) {
    let newHeight = sourceShape.width / targetRatio
    top = (sourceShape.height - newHeight) / 2 / sourceShape.height
    bottom = 1 - top
  }

  // same ratio -> no crop
  else {
    // use the full range as default (from 0 to 1)
  }

  return [top, left, bottom, right]
}

/**
 * @description default is 'rescale'
 *
 * 'rescale' -> scratch/transform to target shape;
 *
 * 'center-crop' -> crop the edges, maintain aspect ratio at center
 */
export type CropAndResizeAspectRatio = 'rescale' | 'center-crop'

/**
 * @description only doing crop and resize
 * - do not auto dispose imageTensor
 * - do not scale the pixel values
 */
export function cropAndResizeImageTensor(options: {
  imageTensor: ImageTensor
  width: number
  height: number
  aspectRatio?: CropAndResizeAspectRatio
}): tf.Tensor4D & tf.Tensor {
  return tf.tidy(() => {
    let { imageTensor, width, height } = options
    let imageShape = getImageTensorShape(imageTensor)
    let cropBox: Box =
      options.aspectRatio != 'center-crop'
        ? [0, 0, 1, 1]
        : calcCropBox({
            sourceShape: imageShape,
            targetShape: { width, height },
          })
    const crop = tf.image.cropAndResize(
      // Expand image input dimensions to add a batch dimension of size 1.
      toTensor4D(imageTensor),
      [cropBox],
      [0],
      [height, width],
    )
    return crop
  })
}

/**
 * @example
 * - [0, 255] -> [0, 1]
 * - [0, 255] -> [-1, 1]
 * - [0, 1] -> [0, 255]
 */
export function scaleTensor(options: {
  tensor: tf.Tensor4D
  fromRange: [min: number, max: number]
  toRange: [min: number, max: number]
}): tf.Tensor4D {
  let { tensor, fromRange, toRange } = options

  // skip mapping if the original range is the same as the target range
  if (fromRange[0] === toRange[0] && fromRange[1] === toRange[1]) {
    return tensor
  }

  let [fromMin, fromMax] = fromRange
  let fromRange = fromMax - fromMin
  let [toMin, toMax] = toRange

  // (x - from_min) / from_range * to_range + to_min
  if (fromMin != 0) {
    tensor = tensor.sub(fromMin)
  }

  return tensor
    .sub<tf.Tensor4D>(fromMin)
    .div<tf.Tensor4D>(fromMax - fromMin)
    .mul<tf.Tensor4D>(toMax - toMin)
    .add<tf.Tensor4D>(toMin)
}
