import { readFile } from 'fs/promises'
import * as tf from '@tensorflow/tfjs-node'
import { readFileSync } from 'fs'

export async function loadImageFileAsync(
  file: string,
  options?: {
    channels?: number
    dtype?: string
    expandAnimations?: boolean
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
  return tensor as tf.Tensor3D
}

export function loadImageFileSync(
  file: string,
  options?: {
    channels?: number
    dtype?: string
    expandAnimations?: boolean
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

export function cropAndResize(options: {
  imageTensor: tf.Tensor3D | tf.Tensor4D
  width: number
  height: number
}): tf.Tensor4D {
  let { imageTensor, width, height } = options
  let imageShape = getImageTensorShape(imageTensor)
  const widthToHeight = imageShape.width / imageShape.height
  let squareCrop: [[number, number, number, number]]
  if (widthToHeight > 1) {
    const heightToWidth = imageShape.height / imageShape.width
    const cropTop = (1 - heightToWidth) / 2
    const cropBottom = 1 - cropTop
    squareCrop = [[cropTop, 0, cropBottom, 1]]
  } else {
    const cropLeft = (1 - widthToHeight) / 2
    const cropRight = 1 - cropLeft
    squareCrop = [[0, cropLeft, 1, cropRight]]
  }
  // Expand image input dimensions to add a batch dimension of size 1.
  let tensor4D =
    imageTensor.shape.length == 4
      ? (imageTensor as tf.Tensor4D)
      : tf.expandDims<tf.Tensor4D>(imageTensor, 1)
  const crop = tf.image.cropAndResize(
    tensor4D,
    squareCrop,
    [0],
    [width, height],
  )
  return crop.div<tf.Tensor4D>(255)
}
