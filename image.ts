import { readFile, writeFile } from 'fs/promises'
import './polyfill'
import * as tf from '@tensorflow/tfjs-node'
import { toTensor3D } from './tensor'
import {
  CropAndResizeAspectRatio,
  cropAndResizeImageTensor,
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
  options?: {
    channels?: number
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
export async function loadImageFile(
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
): Promise<tf.Tensor3D | tf.Tensor4D> {
  let content = await readFile(file)
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
    crop: {
      width: options.width,
      height: options.height,
      aspectRatio: options.aspectRatio,
    },
  })
  let tensor3D = tf.tidy(() => toTensor3D(imageTensor).mul<tf.Tensor3D>(255))
  let content = await tf.node.encodeJpeg(tensor3D)
  tensor3D.dispose()
  await writeFile(options.destFile, content)
}
