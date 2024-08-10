import * as tf from '@tensorflow/tfjs-node'

export function disposeTensor(tensor: tf.Tensor | tf.Tensor[]) {
  if (Array.isArray(tensor)) {
    for (let t of tensor) {
      t.dispose()
    }
  } else {
    tensor.dispose()
  }
}

export function toOneTensor(tensor: tf.Tensor | tf.Tensor[]) {
  if (!Array.isArray(tensor)) {
    return tensor
  }
  if (tensor.length == 0) {
    throw new Error('expect at least one tensor')
  }
  return tensor[0]
}
