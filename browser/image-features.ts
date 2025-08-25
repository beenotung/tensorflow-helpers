import { ImageModel, ImageOrUrl } from './model'
import { Tensor } from '@tensorflow/tfjs-core'
import { getLastSpatialNodeName } from '../spatial-utils'

type tf = {
  tidy: (...args: any[]) => any
}
type image = ImageOrUrl
type node = string | { name: string }

function getName(node: node): string {
  return typeof node == 'string' ? node : node.name
}

export async function getImageFeatures(options: {
  tf: tf
  imageModel: ImageModel
  image: image
  /** default: 'Identity:0' */
  outputNode?: string
  /** default: getLastSpatialNodeName(model) */
  spatialNode?: node
}): Promise<{
  /** e.g. `[1 x 7 x 7 x 160]` */
  spatialFeatures: Tensor
  /** e.g. `[1 x 1280]` */
  pooledFeatures: Tensor
}>
export async function getImageFeatures(options: {
  tf: tf
  imageModel: ImageModel
  image: image
  /** default: 'Identity:0' */
  outputNode?: string
  /** e.g. `imageModel.spatialNodesWithUniqueShapes` */
  spatialNodes: node[]
}): Promise<{
  /**
   * e.g.
   * ```
   * [
   *   [1 x 56 x 56 x 24],
   *   [1 x 28 x 28 x 40],
   *   [1 x 14 x 14 x 80],
   *   [1 x 14 x 14 x 112],
   *   [1 x 7 x 7 x 160],
   * ]
   * ```
   *  */
  spatialFeatures: Tensor[]
  /** e.g. `[1 x 1280]` */
  pooledFeatures: Tensor
}>
export async function getImageFeatures(options: {
  tf: tf
  imageModel: ImageModel
  image: image
  /** default: 'Identity:0' */
  outputNode?: string
  spatialNode?: node
  spatialNodes?: node[]
}): Promise<{
  spatialFeatures: Tensor[] | Tensor
  pooledFeatures: Tensor
}> {
  let { tf, imageModel, image } = options
  let model = imageModel.model
  let spatialNodes =
    options.spatialNodes || options.spatialNode || getLastSpatialNodeName(model)

  let outputNode = options.outputNode || 'Identity:0'
  let names: string[] = Array.isArray(spatialNodes)
    ? [...spatialNodes.map(getName), outputNode]
    : [getName(spatialNodes), outputNode]

  let input = await imageModel.loadImageCropped(image)

  let output = tf.tidy(() => model.execute(input, names) as Tensor[])

  if (typeof image == 'string') {
    input.dispose()
  }

  let spatialFeatures = Array.isArray(spatialNodes)
    ? output.slice(0, output.length - 1)
    : output[0]

  let pooledFeatures = output[output.length - 1]

  return {
    spatialFeatures,
    pooledFeatures,
  }
}
