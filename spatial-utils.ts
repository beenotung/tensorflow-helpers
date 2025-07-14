import { GraphModel } from '@tensorflow/tfjs'

export function getLastSpatialNodeName(model: GraphModel) {
  let names = Object.keys((model as any).executor.graph.nodes)
  // e.g. 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_14/project/BatchNorm/FusedBatchNormV3'
  let spatialNodes = names
    .map(name => {
      let match = name.match(
        /StatefulPartitionedCall\/StatefulPartitionedCall\/predict\/MobilenetV3\/expanded_conv_([\d]+)\/project\/BatchNorm\/FusedBatchNormV3/,
      )
      // starting with 1
      let layer = +match?.[1]!
      return { name, layer }
    })
    .filter(node => node.layer >= 1)
  let lastNode = spatialNodes.sort((a, b) => b.layer - a.layer)[0]
  if (!lastNode) throw new Error('No spatial node found')
  return lastNode.name
}
