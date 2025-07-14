import { ImageModel } from './model'
import { Tensor } from '@tensorflow/tfjs-core'
import { getLastSpatialNodeName } from '../spatial-utils'

export async function getImageFeatures(options: {
  tf: typeof import('@tensorflow/tfjs-core')
  imageModel: ImageModel
  image: string | Tensor
  /** default: 'Identity:0' */
  outputNode?: string
}) {
  let { tf, imageModel, image } = options
  let model = imageModel.model

  let spatialNode = getLastSpatialNodeName(model)
  let outputNode = options.outputNode || 'Identity:0'
  let names = [spatialNode, outputNode]

  let input =
    typeof image == 'string' ? await imageModel.loadImageCropped(image) : image

  // e.g. shape: [1, 7, 7, 160]
  let output = tf.tidy(() => model.execute(input, names) as Tensor[])

  if (typeof image == 'string') {
    input.dispose()
  }

  /** e.g. 7 x 7 x 160 */
  let spatialFeatures = output[0]
  /** e.g. 1 x 1280 */
  let pooledFeatures = output[1]
  return {
    spatialFeatures,
    pooledFeatures,
  }
}
