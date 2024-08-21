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

export function toOneTensor(
  tensor: tf.Tensor | tf.Tensor[] | tf.NamedTensorMap,
): tf.Tensor {
  if (!Array.isArray(tensor)) {
    return tensor as tf.Tensor
  }
  if (tensor.length == 0) {
    throw new Error('expect at least one tensor')
  }
  let result = tf.concat(tensor)
  disposeTensor(tensor)
  return result
}

export function toTensor4D(tensor: tf.Tensor3D | tf.Tensor4D): tf.Tensor4D {
  if (tensor.rank == 4) {
    return tensor as tf.Tensor4D
  }
  let tensor4D = tf.expandDims<tf.Tensor4D>(tensor, 0)
  tensor.dispose()
  return tensor4D
}

export function toTensor3D(tensor: tf.Tensor3D | tf.Tensor4D): tf.Tensor3D {
  if (tensor.rank == 3) {
    return tensor as tf.Tensor3D
  }
  let tensor3D = tensor.squeeze<tf.Tensor3D>([0])
  tensor.dispose()
  return tensor3D
}
