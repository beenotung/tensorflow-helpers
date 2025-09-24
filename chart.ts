import * as tf from '@tensorflow/tfjs'
import { loadImageModel, ModelArtifacts } from './browser'

let chartName = document.querySelector<HTMLElement>('.chart-name')!
let chartNodes = document.querySelector<HTMLElement>('.chart-nodes')!

let canvas = document.querySelector<HTMLCanvasElement>('.node-canvas')!
let context = canvas.getContext('2d')!

let nodeTemplate = chartNodes.querySelector<HTMLElement>('.node')!

nodeTemplate.remove()

type LoadedModelArtifacts = ModelArtifacts & {
  modelTopology: {
    node: Array<ModelTopologyNode>
  }
}
type ModelTopologyNode = {
  name: string
  op: string
  input?: string[]
}

type ModelNode = {
  index: number
  node: ModelTopologyNode
  shape: number[]
  visited: boolean
  inputs: ModelNode[]
  outputs: ModelNode[]
  depth: number
  element: HTMLElement
}

async function main() {
  let model = await loadImageModel({
    url: 'saved_model/mobilenet-v3-large-100',
    cacheUrl: 'indexeddb://mobilenet-v3-large-100',
    checkForUpdates: false,
  })
  console.log('model:', model)
  chartName.textContent = model.spec.url

  let inputNodeName = model.model.inputs[0].name
  console.log('inputNodeName:', inputNodeName)

  let outputNodeName = model.model.outputs[0].name
  console.log('outputNodeName:', outputNodeName)

  let artifacts = (model.model as any).artifacts as LoadedModelArtifacts
  console.log('artifacts:', artifacts)

  let result = model.model.execute(
    tf.zeros([1, 224, 224, 3]),
    artifacts.modelTopology.node.map(node => node.name),
  ) as tf.Tensor[]
  console.log('result:', result)

  let nodes: Record<string, ModelNode> = {}
  artifacts.modelTopology.node.forEach((node, index) => {
    let shape = result[index].shape
    let element = nodeTemplate.cloneNode(true) as HTMLElement
    element.querySelector<HTMLElement>('.node-name')!.textContent = node.name
    element.querySelector<HTMLElement>('.node-op')!.textContent = node.op
    element.querySelector<HTMLElement>('.node-shape')!.textContent =
      shape.join('x')
    nodes[node.name] = {
      index,
      node,
      shape,
      visited: false,
      inputs: [],
      outputs: [],
      depth: 0,
      element,
    }
  })

  // build connections (from output to input)
  let connections: [from: ModelNode, to: ModelNode][] = []
  let stack = [nodes[outputNodeName]]
  Object.values(nodes).forEach(node => {
    node.visited = false
  })
  for (;;) {
    let outputNode = stack.pop()
    if (!outputNode) {
      break
    }
    if (outputNode.visited) {
      continue
    }
    outputNode.visited = true
    // console.log('node:', outputNode.node.name)
    if (outputNode.node.input) {
      for (let nodeName of outputNode.node.input) {
        if (nodeName == '^inputs' && outputNode.node.op == 'Const') {
          continue
        }
        let inputNode = nodes[nodeName]
        inputNode.depth = Math.max(outputNode.depth + 1, inputNode.depth)
        outputNode.inputs.push(inputNode)
        inputNode.outputs.push(outputNode)
        connections.push([inputNode, outputNode])
        stack.push(inputNode)
      }
    }
  }
  console.log('total connections:', connections.length.toLocaleString())

  // build chart (from input to output)
  let nodesArray = Object.values(nodes)
  let maxDepth = Math.max(...nodesArray.map(node => node.depth))
  nodesArray.forEach(node => {
    node.depth = maxDepth - node.depth
    console.log('depth:', node.depth, node.node.name)
  })
  let padding = 32
  let height = 64 * 3.5
  let maxRight = 0
  let maxBottom = 0
  for (let depth = 0; depth <= maxDepth; depth++) {
    let nodes = nodesArray.filter(node => node.depth === depth)
    let left = 0
    for (let node of nodes) {
      // left += node.index
      let element = node.element
      element.querySelector<HTMLElement>('.node-depth')!.textContent =
        node.depth.toString()
      let top = depth * height
      chartNodes.appendChild(element)
      let rect = element.getBoundingClientRect()
      element.style.top = `${top}px`
      element.style.left = `${left}px`
      left += rect.width + padding
      rect = element.getBoundingClientRect()
      maxRight = Math.max(maxRight, rect.right)
      maxBottom = Math.max(maxBottom, rect.bottom)
    }
  }
  canvas.width = maxRight
  canvas.height = maxBottom
  context.clearRect(0, 0, canvas.width, canvas.height)
  let chartRect = chartNodes.getBoundingClientRect()
  for (let [inputNode, outputNode] of connections) {
    let inputRect = inputNode.element.getBoundingClientRect()
    let outputRect = outputNode.element.getBoundingClientRect()
    let from_x = (inputRect.right + inputRect.left) / 2 - chartRect.left
    let to_x = (outputRect.right + outputRect.left) / 2 - chartRect.left
    let from_y = inputRect.bottom - chartRect.top
    let to_y = outputRect.top - chartRect.top
    context.beginPath()
    context.moveTo(from_x, from_y)
    context.lineTo(to_x, to_y)
    context.stroke()
  }
}

main().catch(e => console.error(e))
