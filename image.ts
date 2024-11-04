import * as tf from '@tensorflow/tfjs'
import { toTensor3D } from './tensor'
import {
  CropAndResizeAspectRatio,
  cropAndResizeImageTensor,
} from './image-utils'
import sharp from 'sharp'
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
  const { data, info } = await sharp(file)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true })

  let tensor = tf.tensor4d(
    data,
    [1, info.height, info.width, info.channels],
    'int32',
  )

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
  let tensor3D = tf.tidy(() => toTensor3D(imageTensor)).mul<tf.Tensor3D>(255)
  let shape = tensor3D.shape
  let data = await tensor3D.data()
  tensor3D.dispose()
  await sharp(data, {
    raw: {
      height: shape[0],
      width: shape[1],
      channels: shape[2] as 3,
    },
  }).toFile(options.destFile)
}
