import { readFile, writeFile } from 'fs/promises'
import * as tf from '@tensorflow/tfjs-node'
import { readFileSync } from 'fs'

export async function loadImageFileAsync(
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
  let buffer = await readFile(file)
  let tensor = options
    ? tf.node.decodeImage(
        buffer,
        options.channels,
        options.dtype,
        options.expandAnimations,
      )
    : tf.node.decodeImage(buffer)
  if (options?.crop) {
    tensor = cropAndResize({
      imageTensor: tensor,
      width: options.crop.width,
      height: options.crop.height,
      aspectRatio: options.crop.aspectRatio,
    })
  }
  return tensor as tf.Tensor3D
}

export function loadImageFileSync(
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
): tf.Tensor3D | tf.Tensor4D {
  let buffer = readFileSync(file)
  let tensor = options
    ? tf.node.decodeImage(
        buffer,
        options.channels,
        options.dtype,
        options.expandAnimations,
      )
    : tf.node.decodeImage(buffer)
  if (options?.crop) {
    tensor = cropAndResize({
      imageTensor: tensor,
      width: options.crop.width,
      height: options.crop.height,
      aspectRatio: options.crop.aspectRatio,
    })
  }
  return tensor as tf.Tensor3D
}

export function getImageTensorShape(imageTensor: tf.Tensor3D | tf.Tensor4D) {
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

export function cropAndResize(options: {
  imageTensor: tf.Tensor3D | tf.Tensor4D
  width: number
  height: number
  aspectRatio?: CropAndResizeAspectRatio
}): tf.Tensor4D {
  let { imageTensor, width, height } = options
  let croppedImageTensor = tf.tidy(() => {
    // Expand image input dimensions to add a batch dimension of size 1.
    let imageShape = getImageTensorShape(imageTensor)
    let cropBox: Box =
      options.aspectRatio != 'center-crop'
        ? [0, 0, 1, 1]
        : calcCropBox({
            sourceShape: imageShape,
            targetShape: { width, height },
          })
    let tensor4D =
      imageTensor.shape.length == 4
        ? (imageTensor as tf.Tensor4D)
        : tf.expandDims<tf.Tensor4D>(imageTensor, 0)
    const crop = tf.image.cropAndResize(
      tensor4D,
      [cropBox],
      [0],
      [height, width],
    )
    return crop.div<tf.Tensor4D>(255)
  })
  imageTensor.dispose()
  return croppedImageTensor
}

export async function cropAndResizeImageFileAsync(options: {
  srcFile: string
  destFile: string
  width: number
  height: number
  aspectRatio?: CropAndResizeAspectRatio
}): Promise<void> {
  let imageTensor = await loadImageFileAsync(options.srcFile)
  let tensor3D = tf.tidy(() =>
    cropAndResize({
      imageTensor,
      width: options.width,
      height: options.height,
      aspectRatio: options.aspectRatio,
    })
      .squeeze<tf.Tensor3D>([0])
      .mul<tf.Tensor3D>(255),
  )
  let buffer = Buffer.from(await tf.node.encodeJpeg(tensor3D))
  tensor3D.dispose()
  await writeFile(options.destFile, buffer)
}
