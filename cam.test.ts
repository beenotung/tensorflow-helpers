import * as tf from '@tensorflow/tfjs'
import { loadImageClassifierModel, loadImageModel } from './browser'
import { generate_heatmap_values, heatmap_schemes } from 'heatmap-values'
import { selectImage } from '@beenotung/tslib/file'

console.log('cam.test.ts')

let heatmap_values = generate_heatmap_values(
  heatmap_schemes.red_transparent_blue,
)

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
let camCanvas = document.getElementById('camCanvas') as HTMLCanvasElement
let extractedCanvas = document.getElementById(
  'extractedCanvas',
) as HTMLCanvasElement

let classList = document.querySelector('.class-list') as HTMLDivElement
let outputProbs = document.getElementById('outputProbs') as HTMLDivElement

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

let brightnessRate = 1

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
  if (!clickedImage) {
    return
  }
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
  // crop to center
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

  let r_min = 255
  let r_max = 0

  let g_min = 255
  let g_max = 0

  let b_min = 255
  let b_max = 0

  for (let i = 0; i < imageData.data.length; i += 4) {
    let r = imageData.data[i]
    let g = imageData.data[i + 1]
    let b = imageData.data[i + 2]

    r_min = Math.min(r_min, r)
    r_max = Math.max(r_max, r)
    g_min = Math.min(g_min, g)
    g_max = Math.max(g_max, g)
    b_min = Math.min(b_min, b)
    b_max = Math.max(b_max, b)
  }

  let r_range = r_max - r_min
  let g_range = g_max - g_min
  let b_range = b_max - b_min
  if (r_range == 0 || g_range == 0 || b_range == 0) {
    // too dim or too bright
    requestAnimationFrame(loopVideo)
    return
  }

  let max = Number.NEGATIVE_INFINITY
  let min = Number.POSITIVE_INFINITY

  for (let i = 0; i < imageData.data.length; i += 4) {
    let r = imageData.data[i]
    let g = imageData.data[i + 1]
    let b = imageData.data[i + 2]

    max = Math.max(max, r, g, b)
    min = Math.min(min, r, g, b)

    // r = ((r - r_min) / r_range) * 255
    // g = ((g - g_min) / g_range) * 255
    // b = ((b - b_min) / b_range) * 255

    r = Math.min(r * brightnessRate, 255)
    g = Math.min(g * brightnessRate, 255)
    b = Math.min(b * brightnessRate, 255)

    r = Math.floor(r)
    g = Math.floor(g)
    b = Math.floor(b)

    imageData.data[i] = r
    imageData.data[i + 1] = g
    imageData.data[i + 2] = b
    imageData.data[i + 3] = 255
  }
  inputContext.putImageData(imageData, 0, 0)

  await analyze(imageData)

  requestAnimationFrame(loopVideo)
}

function spreadNormalizedGradientsPeaks(data: number[][]): number[][] {
  let radius = poolingRadius.valueAsNumber
  if (radius <= 0) {
    return data
  }
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
          if (r > radius) {
            continue
          }
          let value = data[ty][tx]
          let dist = Math.abs(value - 0.5)
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
  let selectedClassIndex = -1
  Array.from(classList.children).forEach((button, index) => {
    if (button.classList.contains('active')) {
      selectedClassIndex = index
    }
  })

  let models = await modelsP

  let output = {
    logits: [] as number[],
    probs: [] as number[],
  }
  let gradFunc = tf.grad(x => {
    // x: [1, 224, 224, 3]
    // embedding: [1, 1280]
    let embedding = models.baseModel.model.predict(x) as tf.Tensor
    // logits: [1, 4]
    let logits = models.classifier.classifierModel.predict(
      embedding,
    ) as tf.Tensor
    let probs = tf.softmax(logits)

    output.logits = (logits.arraySync() as number[][])[0]
    output.probs = (probs.arraySync() as number[][])[0]

    // tf.argMax: [1]
    let classIndex: number = selectedClassIndex
    if (classIndex == -1) {
      classIndex = (tf.argMax(logits, -1).arraySync() as number[])[0]
      classList.children[classIndex].classList.add('active')
    }
    let classScore = tf.slice(logits, [0, classIndex], [1, 1])

    return classScore
  })

  let data = tf.tidy(() => {
    let x = tf.browser.fromPixels(inputCanvas)
    x = tf.cast(x, 'float32')
    x = tf.div(x, 255)
    x = tf.expandDims(x, 0)
    let grad = gradFunc(x)
    // [224, 224, 3] -> [224, 224] by taking average of RGB channels
    grad = tf.mean(grad, 3) // mean across channel dimension
    let min = tf.min(grad)
    let max = tf.max(grad)
    let range = tf.sub(max, min)
    // console.log({
    //   min: min.arraySync(),
    //   max: max.arraySync(),
    //   range: range.arraySync(),
    // })
    let normalized = tf.div(tf.sub(grad, min), range)
    let data = normalized.arraySync() as number[][][]
    return data[0]
  })

  data = spreadNormalizedGradientsPeaks(data)

  let maxClassNameLength = Math.max(
    ...models.classifier.classNames.map(className => className.length),
  )

  outputProbs.textContent = models.classifier.classNames
    .map((className, index) => {
      className = className.padStart(maxClassNameLength, ' ')
      return `${className}: ${(output.probs[index] * 100).toFixed(1)}% (${output.logits[index].toFixed(3)})`
    })
    .join('\n')

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
      // Use the same grayscale value for all RGB channels
      imageData.data[i] = color[0]
      imageData.data[i + 1] = color[1]
      imageData.data[i + 2] = color[2]
      imageData.data[i + 3] = color[3] * 255

      let dist = Math.abs(value - 0.5)
      let alpha = Math.floor((dist / 0.5) * 255)
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
  let classifier = await loadImageClassifierModel({
    baseModel,
    modelUrl: 'saved_model/classifier_model',
    cacheUrl: 'indexeddb://classifier-model',
    checkForUpdates: true,
  })

  classList.textContent = ''
  for (let className of classifier.classNames) {
    let button = document.createElement('button')
    button.classList.add('class-item')
    button.textContent = className
    button.onclick = () => {
      if (button.classList.contains('active')) {
        button.classList.remove('active')
        return
      }
      for (let button of classList.children) {
        button.classList.remove('active')
      }
      button.classList.add('active')
      if (clickedImage) {
        analyzeImage(clickedImage)
      }
    }
    classList.appendChild(button)
  }

  return { baseModel, classifier }
}

let modelsP = loadModels()
modelsP.catch(e => {
  console.error(e)
  alert('Failed to load models: ' + String(e))
})

async function startCamera() {
  let stream = await navigator.mediaDevices.getUserMedia({
    video: true,
  })
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

// startCamera().catch(e => console.error(e))
