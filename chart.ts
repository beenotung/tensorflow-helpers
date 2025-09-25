import * as tf from '@tensorflow/tfjs'
import { loadImageModel, ModelArtifacts } from './browser'

let chartName = document.querySelector<HTMLElement>('.chart-name')!
let chartNodes = document.querySelector<HTMLElement>('.chart-nodes')!

let svg = document.querySelector<SVGSVGElement>('.node-svg')!

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
  node: ModelTopologyNode
  inputs: ModelNode[]
  depth: number
  element: HTMLElement
  inputsLines: SVGLineElement[]
  outputsLines: SVGLineElement[]
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
    element.addEventListener('click', () => {
      if (element.classList.contains('locked')) {
        element.classList.remove('locked')
        element.classList.remove('active')
      } else {
        element.classList.add('locked')
      }
    })
    element.addEventListener('mouseover', () => {
      element.classList.add('active')
      for (let line of modelNode.inputsLines) {
        line.classList.add('active')
      }
      for (let line of modelNode.outputsLines) {
        line.classList.add('active')
      }
    })
    element.addEventListener('mouseout', () => {
      if (element.classList.contains('locked')) {
        return
      }
      element.classList.remove('active')
      for (let line of modelNode.inputsLines) {
        line.classList.remove('active')
      }
      for (let line of modelNode.outputsLines) {
        line.classList.remove('active')
      }
    })
    let modelNode: ModelNode = {
      node,
      inputs: [],
      depth: 0,
      element,
      inputsLines: [],
      outputsLines: [],
    }
    nodes[node.name] = modelNode
  })

  // build connections (from output to input)
  let connections: [from: ModelNode, to: ModelNode][] = []
  let visited = new Set<ModelNode>()
  function addConnection(outputNode: ModelNode) {
    if (visited.has(outputNode)) {
      return
    }
    visited.add(outputNode)
    if (outputNode.node.input) {
      for (let nodeName of outputNode.node.input) {
        if (nodeName == '^inputs' && outputNode.node.op == 'Const') {
          continue
        }
        let inputNode = nodes[nodeName]
        if (!inputNode) {
          console.error('inputNode not found:', nodeName)
          continue
        }
        outputNode.inputs.push(inputNode)
        connections.push([inputNode, outputNode])
        addConnection(inputNode)
      }
    }
  }
  addConnection(nodes[outputNodeName])
  console.log('total connections:', connections.length.toLocaleString())

  // calculate depth (from output to input)
  let stack = [nodes[outputNodeName]]
  while (stack.length > 0) {
    let outputNode = stack.shift()!
    for (let inputNode of outputNode.inputs) {
      if (inputNode.depth < outputNode.depth + 1) {
        inputNode.depth = outputNode.depth + 1
        if (!stack.includes(inputNode)) {
          stack.push(inputNode)
        }
      }
    }
  }

  // build chart (from input to output)
  let nodesArray = Object.values(nodes)
  let maxDepth = Math.max(...nodesArray.map(node => node.depth))
  console.log('maxDepth:', maxDepth)
  nodesArray.forEach(node => {
    node.depth = maxDepth - node.depth
    // console.log('depth:', node.depth, node.node.name)
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
  // Set SVG dimensions
  svg.setAttribute('width', maxRight.toString())
  svg.setAttribute('height', maxBottom.toString())
  svg.setAttribute('viewBox', `0 0 ${maxRight} ${maxBottom}`)

  // Clear existing connections
  svg.innerHTML = ''

  // Create connections as SVG lines
  let chartRect = chartNodes.getBoundingClientRect()
  for (let [inputNode, outputNode] of connections) {
    let inputRect = inputNode.element.getBoundingClientRect()
    let outputRect = outputNode.element.getBoundingClientRect()
    let from_x = (inputRect.right + inputRect.left) / 2 - chartRect.left
    let to_x = (outputRect.right + outputRect.left) / 2 - chartRect.left
    let from_y = inputRect.bottom - chartRect.top
    let to_y = outputRect.top - chartRect.top

    let line = document.createElementNS('http://www.w3.org/2000/svg', 'line')
    line.setAttribute('x1', from_x.toString())
    line.setAttribute('y1', from_y.toString())
    line.setAttribute('x2', to_x.toString())
    line.setAttribute('y2', to_y.toString())
    line.setAttribute('stroke', 'black')
    line.setAttribute('stroke-width', '1')
    inputNode.outputsLines.push(line)
    outputNode.inputsLines.push(line)
    svg.appendChild(line)
  }
}

main().catch(e => console.error(e))
