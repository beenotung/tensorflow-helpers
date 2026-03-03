import { readFile, writeFile } from 'fs/promises'
import './polyfill'
import * as tf from '@tensorflow/tfjs-node'
import { toTensor3D } from './tensor'
import {
  CropAndResizeAspectRatio,
  cropAndResizeImageTensor,
  scaleTensor,
} from './image-utils'
export {
  getImageTensorShape,
  Box,
  calcCropBox,
  CropAndResizeAspectRatio,
  cropAndResizeImageTensor,
} from './image-utils'

export async function loadImageFile(
  file: string,
  options: {
    channels?: number
    /** @default [0, 1] */
    outputRange: [min: number, max: number]
    dtype?: string
    expandAnimations?: false
    crop?: {
      width: number
      height: number
      aspectRatio?: CropAndResizeAspectRatio
    }
  },
): Promise<tf.Tensor3D>
export async function loadImageFile(
  file: string,
  options: {
    channels?: number
    /** @default [0, 1] */
    outputRange: [min: number, max: number]
    dtype?: string
    expandAnimations?: boolean
    crop?: {
      width: number
      height: number
      aspectRatio?: CropAndResizeAspectRatio
    }
  },
): Promise<tf.Tensor3D | tf.Tensor4D>
export async function loadImageFile(
  file: string,
  options: {
    channels?: number
    /** @default [0, 1] */
    outputRange: [min: number, max: number]
    dtype?: string
    expandAnimations?: boolean
    crop?: {
      width: number
      height: number
      aspectRatio?: CropAndResizeAspectRatio
    }
  },
): Promise<tf.Tensor3D | tf.Tensor4D> {
  let content = await readFile(file)
  let outputRange = options?.outputRange ?? [0, 1]
  // pixel value range from 0 to 255
  let tensor = options
    ? tf.node.decodeImage(
        content,
        options.channels,
        options.dtype,
        options.expandAnimations ?? false,
      )
    : tf.node.decodeImage(content)
  if (options?.crop) {
    tensor = cropAndResizeImageTensor({
      imageTensor: tensor,
      width: options.crop.width,
      height: options.crop.height,
      aspectRatio: options.crop.aspectRatio,
    })
  }
  if (outputRange) {
    tensor = scaleTensor({
      tensor: tensor as tf.Tensor4D,
      fromRange: [0, 255],
      toRange: outputRange,
    })
  }
  return tensor
}

export async function cropAndResizeImageFile(options: {
  srcFile: string
  destFile: string
  width: number
  height: number
  aspectRatio?: CropAndResizeAspectRatio
}): Promise<void> {
  let imageTensor = await loadImageFile(options.srcFile, {
    range: options.range,
    crop: {
      width: options.width,
      height: options.height,
      aspectRatio: options.aspectRatio,
    },
  })
  let range_min = options.range[0]
  let range_max = options.range[1]
  let tensor3D = tf.tidy(() => {
    if (range_min === 0 && range_max === 1) {
      return toTensor3D(imageTensor).div<tf.Tensor3D>(255)
    }
  })
  let content = await tf.node.encodeJpeg(tensor3D)
  tensor3D.dispose()
  await writeFile(options.destFile, content)
}
