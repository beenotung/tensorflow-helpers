import * as tf from '@tensorflow/tfjs-core'
import { toTensor4D } from './tensor'
import { Sharp } from 'sharp'

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

  let top = 0
  let left = 0
  let bottom = 1
  let right = 1

  if (
    sourceShape.width > sourceShape.height ==
    targetShape.width > targetShape.height
  ) {
    let targetHeightInRatio =
      (sourceShape.height / sourceShape.width) * targetShape.width
    top =
      Math.abs(targetHeightInRatio - targetShape.height) /
      targetHeightInRatio /
      2
    bottom = 1 - top
  } else {
    let targetWidthInRatio =
      (sourceShape.width / sourceShape.height) * targetShape.height
    left =
      Math.abs(targetWidthInRatio - targetShape.width) / targetWidthInRatio / 2
    right = 1 - left
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

export function cropAndResizeImageTensor(options: {
  imageTensor: ImageTensor
  width: number
  height: number
  aspectRatio?: CropAndResizeAspectRatio
}): tf.Tensor4D & tf.Tensor {
  let { imageTensor, width, height } = options
  let croppedImageTensor = tf.tidy(() => {
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
    return tf.div(crop, 255) as tf.Tensor4D
  })
  imageTensor.dispose()
  return croppedImageTensor
}

export function cropAndResizeImageSharp(options: {
  image: Sharp
  width: number
  height: number
  aspectRatio?: CropAndResizeAspectRatio
}): Sharp {
  let { image, width, height } = options

  if (options.aspectRatio != 'center-crop') {
    // scale to target size, stretch to fill
    return image.resize(width, height, { fit: 'fill' })
  } else {
    // crop at center, clip to target size
    return image.resize(width, height, { fit: 'cover', position: 'centre' })
  }
}

export async function imageSharpToTensor(image: Sharp): Promise<tf.Tensor4D> {
  let { data, info } = await image
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true })
  return tf.tidy(() => {
    let tensor = tf.tensor(
      data,
      [info.height, info.width, info.channels],
      'float32',
    )
    tensor = tf.div(tensor, 255)
    tensor = tf.expandDims(tensor, 0)
    return tensor as tf.Tensor4D
  })
}
