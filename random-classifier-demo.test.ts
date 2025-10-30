import * as tf from '@tensorflow/tfjs'
import { loadImageModel, PreTrainedImageModels } from './browser'
import { getImageFeatures } from './image-features'
import { getSpatialNodesWithUniqueShapes } from './spatial-utils'
import { generate_heatmap_values, heatmap_schemes } from 'heatmap-values'
import { selectImage } from '@beenotung/tslib/file'

console.log('random-classifier-demo.test.ts')

let heatmap_values = generate_heatmap_values(
  heatmap_schemes.red_transparent_blue,
)

// Multi-scale layers - will be populated from model
let LAYERS: Array<{
  name: string
  displayName: string
  shape: number[]
  color: string
}> = []

// Create random classifier
function createRandomMultiScaleClassifier() {
  // Calculate total features after concatenation
  let totalFeatures = LAYERS.reduce((sum, layer) => sum + layer.shape[3], 0)

  let classifier = tf.sequential()
  classifier.add(tf.layers.inputLayer({ inputShape: [totalFeatures] }))
  classifier.add(
    tf.layers.dense({
      units: 64,
      activation: 'relu',
      kernelInitializer: 'randomNormal',
      biasInitializer: 'randomNormal',
    }),
  )
  classifier.add(
    tf.layers.dense({
      units: 4, // 4 classes
      activation: 'linear',
      kernelInitializer: 'randomNormal',
      biasInitializer: 'randomNormal',
    }),
  )

  return classifier
}

// Initialize layers from model
function initializeLayersFromModel(baseModel: any) {
  const colors = [
    '#ff6b6b',
    '#4ecdc4',
    '#45b7d1',
    '#96ceb4',
    '#feca57',
    '#ff9ff3',
    '#54a0ff',
  ]

  // Get spatial nodes with unique shapes
  console.log('Getting spatial nodes from model...')
  let spatialNodes = getSpatialNodesWithUniqueShapes({
    model: baseModel.model,
    tf: tf,
  })
  console.log('Found spatial nodes:', spatialNodes.length)
  console.log('Spatial nodes:', spatialNodes)

  // Filter to get the layers we want (every other one to get good variety)
  let selectedNodes = spatialNodes
    .filter((_, index) => index % 2 === 0)
    .slice(0, 5)
  console.log('Selected nodes:', selectedNodes.length)

  LAYERS = selectedNodes.map((node, index) => ({
    name: node.name,
    displayName: `Layer ${node.layer} (${node.shape[1]}×${node.shape[2]})`,
    shape: node.shape,
    color: colors[index % colors.length],
  }))

  console.log('Initialized layers:', LAYERS)
}

let poolingRadius = document.getElementById('poolingRadius') as HTMLInputElement
let catImage = document.getElementById('catImage') as HTMLImageElement
let dogImage = document.getElementById('dogImage') as HTMLImageElement
let pickImageButton = document.getElementById(
  'pickImageButton',
) as HTMLButtonElement
let inputCanvas = document.getElementById('inputCanvas') as HTMLCanvasElement
let camCanvas = document.getElementById('camCanvas') as HTMLCanvasElement
let extractedCanvas = document.getElementById(
  'extractedCanvas',
) as HTMLCanvasElement

let inputContext = inputCanvas.getContext('2d')!
let camContext = camCanvas.getContext('2d')!
let extractedContext = extractedCanvas.getContext('2d')!

let size = 224
inputCanvas.width = size
inputCanvas.height = size
camCanvas.width = size
camCanvas.height = size
extractedCanvas.width = size
extractedCanvas.height = size

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
  let randomClassifier = models.randomClassifier

  let output = {
    logits: [] as number[],
    probs: [] as number[],
  }

  // Multi-scale CAM with random classifier
  let data: number[][]

  // Extract features outside of tf.tidy
  let x = tf.browser.fromPixels(inputCanvas)
  x = tf.cast(x, 'float32')
  x = tf.div(x, 255)
  x = tf.expandDims(x, 0)

  // Extract multi-scale features directly
  let spatialNodeNames = LAYERS.map(layer => layer.name)
  let spatialFeatures: tf.Tensor[] = []

  console.log('Available layers:', LAYERS.length)
  console.log('Layer names:', spatialNodeNames)

  for (let nodeName of spatialNodeNames) {
    try {
      console.log(`Attempting to extract from: ${nodeName}`)
      let layerOutput = models.baseModel.model.execute(x, [
        nodeName,
      ]) as tf.Tensor[]
      console.log(`Layer output:`, layerOutput)
      if (layerOutput && layerOutput[0]) {
        console.log(
          `Successfully extracted features with shape:`,
          layerOutput[0].shape,
        )
        spatialFeatures.push(layerOutput[0])
      } else {
        console.warn(`No output from layer: ${nodeName}`)
      }
    } catch (error) {
      console.warn(`Failed to extract features from ${nodeName}:`, error)
    }
  }

  data = tf.tidy(() => {
    let layerFeatures = spatialFeatures as tf.Tensor[]
    console.log('Extracted features from', layerFeatures.length, 'layers')

    // Concatenate all features (upsampled to same resolution)
    let upsampledFeatures = layerFeatures.map((features, index) => {
      try {
        console.log(`Upsampling layer ${index}, shape:`, features.shape)
        // Ensure the tensor has the right rank and shape
        if (features.rank !== 4) {
          console.warn(`Layer ${index} has rank ${features.rank}, expected 4`)
          return tf.zeros([1, 224, 224, 1])
        }

        // Check if the spatial dimensions are valid
        let [batch, height, width, channels] = features.shape
        if (height <= 0 || width <= 0 || channels <= 0) {
          console.warn(`Layer ${index} has invalid dimensions:`, features.shape)
          return tf.zeros([1, 224, 224, 1])
        }

        return tf.image.resizeBilinear(features as tf.Tensor3D, [224, 224])
      } catch (error) {
        console.error(`Failed to upsample layer ${index}:`, error)
        return tf.zeros([1, 224, 224, 1])
      }
    })

    // Filter out any failed upsamplings
    upsampledFeatures = upsampledFeatures.filter(features => features != null)

    if (upsampledFeatures.length === 0) {
      console.error('No valid features to concatenate')
      return Array(224)
        .fill(null)
        .map(() => Array(224).fill(0.5))
    }

    let concatenated = tf.concat(upsampledFeatures, 3) // [1, 224, 224, total_channels]

    // Global average pooling
    let pooled = tf.mean(concatenated, [1, 2]) // [1, total_channels]

    // Get classification results
    let logits = randomClassifier.predict(pooled) as tf.Tensor
    let probs = tf.softmax(logits)

    let logitsArray = logits.arraySync() as number[][]
    let probsArray = probs.arraySync() as number[][]

    output.logits = logitsArray[0] || []
    output.probs = probsArray[0] || []

    // Compute gradients for CAM
    let gradFunc = tf.grad((input: tf.Tensor) => {
      try {
        // Forward pass through multi-scale features
        let layerFeatures: tf.Tensor[] = []
        LAYERS.forEach((layer, index) => {
          try {
            let layerOutput = models.baseModel.model.execute(input, [
              layer.name,
            ]) as tf.Tensor[]
            let features = layerOutput[0]

            // Validate shape before using
            if (
              features &&
              features.rank === 4 &&
              features.shape &&
              features.shape.length >= 3 &&
              features.shape[1] != null &&
              features.shape[2] != null &&
              features.shape[1] > 0 &&
              features.shape[2] > 0
            ) {
              layerFeatures.push(features)
            } else {
              console.warn(
                `Invalid features for layer ${index}:`,
                features?.shape || 'undefined',
              )
              layerFeatures.push(tf.zeros([1, 224, 224, 1]))
            }
          } catch (error) {
            console.warn(`Failed to get features for layer ${index}:`, error)
            layerFeatures.push(tf.zeros([1, 224, 224, 1]))
          }
        })

        // Safe upsampling
        let upsampledFeatures = layerFeatures.map((features, index) => {
          try {
            if (
              features &&
              features.rank === 4 &&
              features.shape &&
              features.shape.length >= 3 &&
              features.shape[1] != null &&
              features.shape[2] != null &&
              features.shape[1] > 0 &&
              features.shape[2] > 0
            ) {
              return tf.image.resizeBilinear(
                features as tf.Tensor3D,
                [224, 224],
              )
            } else {
              return tf.zeros([1, 224, 224, 1])
            }
          } catch (error) {
            console.warn(`Failed to upsample layer ${index}:`, error)
            return tf.zeros([1, 224, 224, 1])
          }
        })

        let concatenated = tf.concat(upsampledFeatures, 3)
        let pooled = tf.mean(concatenated, [1, 2])
        let logits = randomClassifier.predict(pooled) as tf.Tensor

        // Return the highest scoring class
        return tf.max(logits, 1)
      } catch (error) {
        console.error('Error in gradient computation:', error)
        return tf.zeros([1])
      }
    })

    try {
      let grad = gradFunc(x)
      grad = tf.mean(grad, 3) // mean across channel dimension
      let min = tf.min(grad)
      let max = tf.max(grad)
      let range = tf.sub(max, min)
      let normalized = tf.div(tf.sub(grad, min), range)
      let gradData = normalized.arraySync() as number[][][]
      return gradData[0]
    } catch (error) {
      console.error('Error computing gradients:', error)
      return Array(224)
        .fill(null)
        .map(() => Array(224).fill(0.5))
    }
  })

  data = spreadNormalizedGradientsPeaks(data, {
    radius: poolingRadius.valueAsNumber,
    spread_positive: true,
    spread_negative: true,
  })

  // Display results
  let outputProbs = document.getElementById('outputProbs') as HTMLDivElement
  outputProbs.textContent = `Random Classifier Results:
Class 0: ${(output.probs[0] * 100).toFixed(1)}% (${output.logits[0].toFixed(3)})
Class 1: ${(output.probs[1] * 100).toFixed(1)}% (${output.logits[1].toFixed(3)})
Class 2: ${(output.probs[2] * 100).toFixed(1)}% (${output.logits[2].toFixed(3)})
Class 3: ${(output.probs[3] * 100).toFixed(1)}% (${output.logits[3].toFixed(3)})

Note: These are random predictions, but the spatial features are still meaningful!`

  // Visualize CAM
  let i = 0
  for (let y = 0; y < imageData.height; y++) {
    for (let x = 0; x < imageData.width; x++, i += 4) {
      let value = data[y][x]
      let dist = Math.abs(value - 0.5)
      let alpha = Math.floor((dist / 0.5) * 255)
      imageData.data[i + 3] = alpha
    }
  }
  extractedContext.putImageData(imageData, 0, 0)

  i = 0
  for (let y = 0; y < imageData.height; y++) {
    for (let x = 0; x < imageData.width; x++, i += 4) {
      let value = data[y][x]
      let index = Math.floor(value * 255)
      let color = heatmap_values[index]
      imageData.data[i] = color[0]
      imageData.data[i + 1] = color[1]
      imageData.data[i + 2] = color[2]
      imageData.data[i + 3] = color[3] * 255
    }
  }
  camContext.putImageData(imageData, 0, 0)
}

async function loadModels() {
  let baseModel = await loadImageModel({
    url: 'saved_model/base_model',
    cacheUrl: 'indexeddb://base-model',
    checkForUpdates: false,
  })

  // Initialize layers from the model
  initializeLayersFromModel(baseModel)

  // Create random classifier
  let randomClassifier = createRandomMultiScaleClassifier()
  console.log('Random multi-scale classifier created')

  return { baseModel, randomClassifier }
}

let modelsP = loadModels()
modelsP.catch(e => {
  console.error(e)
  alert('Failed to load models: ' + String(e))
})
