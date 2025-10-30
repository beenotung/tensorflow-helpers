import * as tf from '@tensorflow/tfjs'
import {
  getSpatialNodesWithUniqueShapes,
  loadImageModel,
  PreTrainedImageModels,
} from './browser'
import { generate_heatmap_values, heatmap_schemes } from 'heatmap-values'
import { selectImage } from '@beenotung/tslib/file'

console.log('multi-scale-visualization.test.ts')

let heatmap_values = generate_heatmap_values(
  heatmap_schemes.red_transparent_blue,
)

// Multi-scale layer configuration
const MULTI_SCALE_LAYERS = [
  {
    name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_2/project/BatchNorm/FusedBatchNormV3',
    displayName: 'Layer 2 (56×56)',
    expectedShape: [1, 56, 56, 24],
    color: '#ff6b6b', // Red
  },
  {
    name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_5/project/BatchNorm/FusedBatchNormV3',
    displayName: 'Layer 5 (28×28)',
    expectedShape: [1, 28, 28, 40],
    color: '#4ecdc4', // Teal
  },
  {
    name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_9/project/BatchNorm/FusedBatchNormV3',
    displayName: 'Layer 9 (14×14)',
    expectedShape: [1, 14, 14, 80],
    color: '#45b7d1', // Blue
  },
  {
    name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_11/project/BatchNorm/FusedBatchNormV3',
    displayName: 'Layer 11 (14×14)',
    expectedShape: [1, 14, 14, 112],
    color: '#96ceb4', // Green
  },
  {
    name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_14/project/BatchNorm/FusedBatchNormV3',
    displayName: 'Layer 14 (7×7)',
    expectedShape: [1, 7, 7, 160],
    color: '#feca57', // Yellow
  },
]

let poolingRadius = document.getElementById('poolingRadius') as HTMLInputElement
let catImage = document.getElementById('catImage') as HTMLImageElement
let dogImage = document.getElementById('dogImage') as HTMLImageElement
let pickImageButton = document.getElementById(
  'pickImageButton',
) as HTMLButtonElement
let startCameraButton = document.getElementById(
  'startCameraButton',
) as HTMLButtonElement
let startVideoButton = document.getElementById(
  'startVideoButton',
) as HTMLButtonElement
let video = document.getElementById('video') as HTMLVideoElement
let inputCanvas = document.getElementById('inputCanvas') as HTMLCanvasElement

// Create canvas elements for each layer
let layerCanvases: HTMLCanvasElement[] = []
let layerContexts: CanvasRenderingContext2D[] = []

let inputContext = inputCanvas.getContext('2d')!
let size = 224
inputCanvas.width = size
inputCanvas.height = size

let brightnessRate = 1

// Initialize layer canvases
function initializeLayerCanvases() {
  const container = document.getElementById('layerContainer')!

  MULTI_SCALE_LAYERS.forEach((layer, index) => {
    // Create layer container
    const layerDiv = document.createElement('div')
    layerDiv.className = 'layer-container'
    layerDiv.innerHTML = `
      <h3>${layer.displayName}</h3>
      <canvas id="layer${index}Canvas" width="224" height="224"></canvas>
      <div class="layer-info">
        <span>Shape: ${layer.expectedShape.join('×')}</span>
        <span>Color: <span style="color: ${layer.color}">●</span></span>
      </div>
    `
    container.appendChild(layerDiv)

    // Get canvas and context
    const canvas = document.getElementById(
      `layer${index}Canvas`,
    ) as HTMLCanvasElement
    const context = canvas.getContext('2d')!
    layerCanvases.push(canvas)
    layerContexts.push(context)
  })
}

startCameraButton.onclick = async () => {
  brightnessRate = 4
  video.controls = false
  await startCamera()
}

startVideoButton.onclick = async () => {
  brightnessRate = 1
  video.controls = true
  await startVideo()
}

let clickedImage = null as HTMLImageElement | null
catImage.onclick = async () => {
  clickedImage = catImage
  await analyzeImage(clickedImage)
}
dogImage.onclick = async () => {
  clickedImage = dogImage
  await analyzeImage(clickedImage)
}
pickImageButton.onclick = async () => {
  let [file] = await selectImage({ accept: 'image/*' })
  if (!file) return
  let image = new Image()
  await new Promise<void>(resolve => {
    image.onload = () => resolve()
    image.src = URL.createObjectURL(file)
  })
  clickedImage = image
  await analyzeImage(clickedImage)
}

poolingRadius.onchange = async () => {
  if (!clickedImage) return
  await analyzeImage(clickedImage)
}

async function analyzeImage(image: HTMLImageElement) {
  let inputSize = Math.min(image.naturalWidth, image.naturalHeight)
  let left = (image.naturalWidth - inputSize) / 2
  let top = (image.naturalHeight - inputSize) / 2
  inputContext.drawImage(
    image,
    left,
    top,
    inputSize,
    inputSize,
    0,
    0,
    inputCanvas.width,
    inputCanvas.height,
  )
  let imageData = inputContext.getImageData(
    0,
    0,
    inputCanvas.width,
    inputCanvas.height,
  )
  await analyze(imageData)
}

async function loopVideo() {
  let inputSize = Math.min(video.videoWidth, video.videoHeight)
  let left = (video.videoWidth - inputSize) / 2
  let top = (video.videoHeight - inputSize) / 2
  inputContext.drawImage(
    video,
    left,
    top,
    inputSize,
    inputSize,
    0,
    0,
    inputCanvas.width,
    inputCanvas.height,
  )
  let imageData = inputContext.getImageData(
    0,
    0,
    inputCanvas.width,
    inputCanvas.height,
  )

  // Brightness adjustment
  for (let i = 0; i < imageData.data.length; i += 4) {
    imageData.data[i] = Math.min(imageData.data[i] * brightnessRate, 255)
    imageData.data[i + 1] = Math.min(
      imageData.data[i + 1] * brightnessRate,
      255,
    )
    imageData.data[i + 2] = Math.min(
      imageData.data[i + 2] * brightnessRate,
      255,
    )
    imageData.data[i + 3] = 255
  }
  inputContext.putImageData(imageData, 0, 0)

  await analyze(imageData)
  requestAnimationFrame(loopVideo)
}

function spreadNormalizedGradientsPeaks(
  data: number[][],
  options: {
    radius: number
    spread_positive?: boolean
    spread_negative?: boolean
  },
): number[][] {
  let { radius, spread_positive, spread_negative } = options
  if (radius <= 0) return data

  let newData = data.map(row => new Array(row.length))
  for (let y = 0; y < data.length; y++) {
    let min_y = Math.max(y - radius, 0)
    let max_y = Math.min(y + radius, data.length - 1)
    for (let x = 0; x < data[y].length; x++) {
      let min_x = Math.max(x - radius, 0)
      let max_x = Math.min(x + radius, data[y].length - 1)
      let newValue = 0.5
      let newDist = 0
      for (let ty = min_y; ty <= max_y; ty++) {
        let dy = ty - y
        for (let tx = min_x; tx <= max_x; tx++) {
          let dx = tx - x
          let r = Math.sqrt(dx * dx + dy * dy)
          if (r > radius) continue
          let value = data[ty][tx]
          let dist: number
          if (spread_positive && spread_negative) {
            dist = Math.abs(value - 0.5)
          } else if (spread_positive) {
            dist = value - 0.5
          } else if (spread_negative) {
            dist = 0.5 - value
          } else {
            throw new Error(
              'either spread_positive or spread_negative must be true',
            )
          }
          if (dist > newDist) {
            newValue = value
            newDist = dist
          }
        }
      }
      newData[y][x] = newValue
    }
  }
  return newData
}

async function analyze(imageData: ImageData) {
  let models = await modelsP

  let data = tf.tidy(() => {
    let x = tf.browser.fromPixels(inputCanvas)
    x = tf.cast(x, 'float32')
    x = tf.div(x, 255)
    x = tf.expandDims(x, 0)

    // Extract features from all layers
    let layerFeatures: tf.Tensor[] = []

    MULTI_SCALE_LAYERS.forEach((layer, index) => {
      try {
        let layerOutput = models.baseModel.model.execute(x, [
          layer.name,
        ]) as tf.Tensor[]
        let spatialFeatures = layerOutput[0]
        console.log(`${layer.displayName} shape:`, spatialFeatures.shape)
        layerFeatures.push(spatialFeatures)
      } catch (error) {
        console.error(`Failed to get ${layer.displayName}:`, error)
        // Create dummy tensor with expected shape
        let dummyShape = layer.expectedShape
        layerFeatures.push(tf.zeros(dummyShape))
      }
    })

    return layerFeatures
  })

  // Visualize each layer
  data.forEach((spatialFeatures, index) => {
    let layer = MULTI_SCALE_LAYERS[index]
    let canvas = layerCanvases[index]
    let context = layerContexts[index]

    // Process spatial features
    let processed = tf.tidy(() => {
      // Average across channels
      let channelAveraged = tf.mean(spatialFeatures, 3, true) // [1, H, W, 1]

      // Upsample to 224x224
      let upsampled = tf.image.resizeBilinear(
        channelAveraged as tf.Tensor3D,
        [224, 224],
      ) // [1, 224, 224, 1]

      let squeezed = tf.squeeze(upsampled, [0, 3]) // [224, 224]

      // Normalize
      let min = tf.min(squeezed)
      let max = tf.max(squeezed)
      let range = tf.sub(max, min)
      let normalized = tf.div(tf.sub(squeezed, min), range)

      return normalized.arraySync() as number[][]
    })

    // Apply smoothing
    processed = spreadNormalizedGradientsPeaks(processed, {
      radius: poolingRadius.valueAsNumber,
      spread_positive: true,
      spread_negative: true,
    })

    // Create visualization
    let canvasData = context.createImageData(224, 224)
    let i = 0
    for (let y = 0; y < 224; y++) {
      for (let x = 0; x < 224; x++, i += 4) {
        let value = processed[y][x]
        let index = Math.floor(value * 255)
        let color = heatmap_values[index]

        // Apply layer-specific color tint
        let r = Math.floor(
          color[0] * 0.7 + parseInt(layer.color.slice(1, 3), 16) * 0.3,
        )
        let g = Math.floor(
          color[1] * 0.7 + parseInt(layer.color.slice(3, 5), 16) * 0.3,
        )
        let b = Math.floor(
          color[2] * 0.7 + parseInt(layer.color.slice(5, 7), 16) * 0.3,
        )

        canvasData.data[i] = r
        canvasData.data[i + 1] = g
        canvasData.data[i + 2] = b
        canvasData.data[i + 3] = color[3] * 255
      }
    }

    context.putImageData(canvasData, 0, 0)
  })

  // Clean up
  data.forEach(tensor => tensor.dispose())
}

async function loadModels() {
  let baseModel = await loadImageModel({
    url: 'saved_model/base_model',
    cacheUrl: 'indexeddb://base-model',
    checkForUpdates: false,
  })

  console.log('Model loaded successfully')
  return { baseModel }
}

let modelsP = loadModels()
modelsP.catch(e => {
  console.error(e)
  alert('Failed to load models: ' + String(e))
})

async function startCamera() {
  let stream = await navigator.mediaDevices.getUserMedia({ video: true })
  await new Promise<void>(resolve => {
    video.onloadedmetadata = () => resolve()
    video.srcObject = stream
    video.play()
  })
  loopVideo()
}

async function startVideo() {
  await new Promise<void>(resolve => {
    video.onloadedmetadata = () => resolve()
    let file =
      'A dog, a cat, and a hamster who gets the last treat [3F5Cz_YN8FA].mp4'
    video.src = 'samples/' + file
    video.play()
  })
  loopVideo()
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
  initializeLayerCanvases()
})
