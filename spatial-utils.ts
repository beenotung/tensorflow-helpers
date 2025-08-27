import type { GraphModel, Tensor } from '@tensorflow/tfjs'

export function getSpatialNodeNames(model: GraphModel) {
  let names = Object.keys((model as any).executor.graph.nodes)
  // e.g. 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_14/project/BatchNorm/FusedBatchNormV3'
  let spatialNodes = names
    .map(name => {
      let match = name.match(
        /StatefulPartitionedCall\/StatefulPartitionedCall\/predict\/MobilenetV3\/expanded_conv_([\d]+)\/project\/BatchNorm\/FusedBatchNormV3/,
      )
      // starting from 1
      let layer = +match?.[1]!
      return { name, layer }
    })
    .filter(node => node.layer >= 1)
  return spatialNodes
}

export function getLastSpatialNodeName(model: GraphModel) {
  let spatialNodes = getSpatialNodeNames(model)
  let lastNode = spatialNodes.sort((a, b) => b.layer - a.layer)[0]
  if (!lastNode) throw new Error('No spatial node found')
  return lastNode.name
}

export type SpatialNode = {
  name: string
  layer: number
  shape: number[]
}

export function getSpatialNodes(args: {
  model: GraphModel
  tf: { zeros: (shape: number[]) => Tensor }
}): SpatialNode[] {
  let { model, tf } = args
  let spatialNodes = getSpatialNodeNames(model)
  let shape = model.inputs[0].shape!.slice(0)
  shape[0] = 1
  let input = tf.zeros(shape)
  let output = model.execute(
    input,
    spatialNodes.map(node => node.name),
  ) as Tensor[]
  return output.map((tensor, i) => {
    let { name, layer } = spatialNodes[i]
    return {
      name,
      layer,
      shape: tensor.shape,
    }
  })
}

export function filterSpatialNodesWithUniqueShapes(
  spatialNodes: SpatialNode[],
) {
  return spatialNodes
    .slice()
    .sort((a, b) => a.layer - b.layer)
    .filter((current, i, nodes) => {
      let next = nodes[i + 1]
      if (!next) return true
      let current_shape = current.shape.join()
      let next_shape = next.shape.join()
      return current_shape != next_shape
    })
}

export function getSpatialNodesWithUniqueShapes(args: {
  model: GraphModel
  tf: { zeros: (shape: number[]) => Tensor }
}) {
  let { model, tf } = args
  let spatialNodes = getSpatialNodes({ model, tf })
  spatialNodes = filterSpatialNodesWithUniqueShapes(spatialNodes)
  return spatialNodes
}
