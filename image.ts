import * as tf from '@tensorflow/tfjs'
import {
  CropAndResizeAspectRatio,
  cropAndResizeImageSharp,
  cropAndResizeImageTensor,
  imageSharpToTensor,
} from './image-utils'
import sharp from 'sharp'
export {
  getImageTensorShape,
  Box,
  calcCropBox,
  CropAndResizeAspectRatio,
  cropAndResizeImageSharp,
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
  let image = sharp(file)
  if (options?.crop) {
    image = cropAndResizeImageSharp({
      image,
      ...options.crop,
    })
  }
  return imageSharpToTensor(image)
}

export async function cropAndResizeImageFile(options: {
  srcFile: string
  destFile: string
  width: number
  height: number
  aspectRatio?: CropAndResizeAspectRatio
}): Promise<void> {
  let image = sharp(options.srcFile)
  image = cropAndResizeImageSharp({
    image,
    width: options.width,
    height: options.height,
    aspectRatio: options.aspectRatio,
  })
  await image.toFile(options.destFile)
}
