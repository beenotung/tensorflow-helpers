import { fileToBase64String } from '@beenotung/tslib/file'
import {
  calcHiddenLayerSize,
  getImageFeatures,
  loadImageModel,
  loadLayersModel,
  PreTrainedImageModels,
} from '../browser'
import * as tf from '@tensorflow/tfjs'
import { standardizeClassWeights } from '@tensorflow/tfjs-layers/dist/engine/training_utils'

let input = document.getElementById('input') as HTMLInputElement
let image = document.getElementById('image') as HTMLImageElement
let canvas = document.getElementById('canvas') as HTMLCanvasElement
let context = canvas.getContext('2d')!

let spec = PreTrainedImageModels.mobilenet['mobilenet-v3-large-100']
// 1x7x7x160
let ROW = spec.spatial_features[1]
let COL = spec.spatial_features[2]

async function loadModels() {
  /* image model: 224x224x3 -> 7x7x160 */
  let baseModel = await loadImageModel({
    url: './saved_models/base_model/model.json',
    // url: spec.url,
    cacheUrl: 'indexeddb://mobilenet-v3-large-100',
    checkForUpdates: false,
  })

  let hiddenLayerSize = 32
  let outputSize = 2

  let classifierModel = tf.sequential()
  /* classifier model: 160 features -> 2 (background or object) */
  classifierModel.add(
    tf.layers.inputLayer({
      inputShape: [spec.spatial_features.slice().pop()!],
    }),
  )
  classifierModel.add(tf.layers.dropout({ rate: 0.5 }))
  /* hidden layer */
  classifierModel.add(
    tf.layers.dense({ units: hiddenLayerSize, activation: 'gelu' }),
  )
  classifierModel.add(tf.layers.dropout({ rate: 0.5 }))
  /* output layer */
  classifierModel.add(tf.layers.dense({ units: 2, activation: 'softmax' }))

  /*2d*/
  let input = tf.input({ shape: spec.spatial_features.slice(1) })
  let hidden = tf.layers
    .conv2d({
      filters: hiddenLayerSize,
      kernelSize: 1,
      activation: 'gelu',
    })
    .apply(input, ['spatial_features_to_hidden']) as tf.SymbolicTensor
  let output = tf.layers
    .conv2d({
      filters: outputSize,
      kernelSize: 1,
      activation: 'softmax',
    })
    .apply(hidden, ['hidden_to_output']) as tf.SymbolicTensor
  let classifierModel2D = tf.model({ inputs: input, outputs: output })

  return { baseModel, classifierModel, classifierModel2D }
}

let modelsPromise = loadModels()

async function checkFile() {
  let file = input.files?.[0]
  if (!file) return
  console.log(file)
  let dataUrl = await fileToBase64String(file)
  await new Promise(resolve => {
    image.onload = resolve
    image.src = dataUrl
  })
  let filename = file.name
  let filePath = './dataset/test/labels/' + filename.replace('.jpg', '.txt')
  let data
  try {
    const response = await fetch(filePath)
    if (!response.ok) {
      throw new Error(`Failed to fetch ${filePath}: ${response.statusText}`)
    }
    data = await response.text()
  } catch (error) {
    console.error(error)
  }
  canvas.width = image.naturalWidth
  canvas.height = image.naturalHeight
  context.drawImage(image, 0, 0)

  let boxHeight = canvas.height / ROW
  let boxWidth = canvas.width / COL

  let expected_box: NormalizedBox = {
    x: data ? Number(data.split(' ')[1]) : 0,
    y: data ? Number(data.split(' ')[2]) : 0,
    width: data ? Number(data.split(' ')[3]) : 0,
    height: data ? Number(data.split(' ')[4]) : 0,
  }
  let expected_box_width = expected_box.width * canvas.width
  let expected_box_height = expected_box.height * canvas.height
  let expected_box_left = expected_box.x * canvas.width - expected_box_width / 2
  let expected_box_top =
    expected_box.y * canvas.height - expected_box_height / 2
  let expected_box_right = expected_box_left + expected_box_width
  let expected_box_bottom = expected_box_top + expected_box_height
  let expected_box_pixel: PixelBox = {
    left: expected_box_left,
    top: expected_box_top,
    right: expected_box_right,
    bottom: expected_box_bottom,
  }

  let { baseModel } = await modelsPromise
  let classifierModel = await loadLayersModel({
    url: './saved_models/c1-model',
  })
  let embedding = await getImageFeatures({
    tf,
    imageModel: baseModel,
    image: dataUrl,
  })
  let spatialFeatures = embedding.spatialFeatures
  console.log({ spatialFeatures })

  // compile()
  // await train({ epoch: 100 })
  await predict()

  async function predict() {
    let result2D = classifierModel.predict(spatialFeatures) as tf.Tensor
    console.log({ result2D })
    // 1x7x7x2
    let [matrix] = (await result2D.array()) as number[][][][]

    let targets = [] as [x: number, y: number][]
    const target_threshold = 0.3

    /* draw heatmap */
    for (let y = 0; y < matrix.length; y++) {
      for (let x = 0; x < matrix[y].length; x++) {
        let [background, object] = matrix[y][x]
        if (object >= target_threshold) {
          targets.push([x, y])
        }
        let opacity = object * 0.5
        drawBox({
          box: {
            x: x / COL + boxWidth / canvas.width / 2,
            y: y / ROW + boxHeight / canvas.height / 2,
            width: boxWidth / canvas.width,
            height: boxHeight / canvas.height,
          },
          borderColor: '#eeeeee55',
          fillColor: `rgba(255,0,0,${opacity})`,
        })
      }
    }
    // TODO calculate it from the heatmap
    let target_x = targets.map((t) => t[0])
    let target_y = targets.map((t) => t[1])

    let predicted_box: NormalizedBox = {
      x: (Math.min(...target_x) + Math.max(...target_x) + 1) / (2 * COL),
      y: (Math.min(...target_y) + Math.max(...target_y) + 1) / (2 * ROW),
      width: (Math.max(...target_x) - Math.min(...target_x) + 1) / COL,
      height: (Math.max(...target_y) - Math.min(...target_y) + 1) / ROW,
    }

    /* draw bounding box */
    drawBox({
      box: expected_box,
      borderColor: '#00ff00',
      fillColor: 'transparent',
    })
    drawBox({
      box: predicted_box,
      borderColor: '#ff0000',
      fillColor: 'transparent',
    })
  }

  /* setup for training */
  // function compile() {
  //   classifierModel2D.compile({
  //     optimizer: 'adam',
  //     loss: tf.metrics.categoricalCrossentropy,
  //     metrics: [tf.metrics.categoricalAccuracy],
  //   })
  // }
  /* train the model */
  // async function train(options: { epoch: number }) {
  //   console.log('before train')

  //   let background_count = 0
  //   let object_count = 0

  //   let ys = []
  //   for (let y = 0 y < ROW y++) {
  //     let xs = []
  //     for (let x = 0 x < COL x++) {
  //       let grid: PixelBox = {
  //         left: x * boxWidth,
  //         top: y * boxHeight,
  //         right: (x + 1) * boxWidth,
  //         bottom: (y + 1) * boxHeight,
  //       }
  //       let score = calcGridScore(grid, expected_box_pixel)
  //       let object = score
  //       let background = 1 - object
  //       xs.push([background, object])
  //       background_count += background
  //       object_count += object
  //     }
  //     ys.push(xs)
  //   }
  //   console.log({ background_count, object_count })
  //   let total_count = background_count + object_count
  //   await classifierModel2D.fit(spatialFeatures, tf.tensor([ys]), {
  //     epochs: options.epoch,
  //     // classWeight: [
  //     //   (1 - background_count / total_count) * 2,
  //     //   (1 - object_count / total_count) * 2,
  //     // ],
  //   })
  //   console.log('after train')
  // }
}

type NormalizedBox = {
  x: number
  y: number
  width: number
  height: number
}

type PixelBox = {
  left: number
  top: number
  right: number
  bottom: number
}

function toPixelBox(box: NormalizedBox): PixelBox {
  let width = box.width * canvas.width
  let height = box.height * canvas.height
  let left = box.x * canvas.width - width / 2
  let top = box.y * canvas.height - height / 2
  let right = left + width
  let bottom = top + height
  return {
    left,
    top,
    right,
    bottom,
  }
}

function toNormalizedBox(box: PixelBox): NormalizedBox {
  let left = box.left / canvas.width
  let right = box.right / canvas.width
  let top = box.top / canvas.height
  let bottom = box.bottom / canvas.height
  let width = right - left
  let height = bottom - top
  let x = (left + right) / 2
  let y = (top + bottom) / 2
  return {
    x,
    y,
    width,
    height,
  }
}

let grid_score_threshold = 1 / 3

/* output 0 or 1 exactly */
function calcGridScore(grid: PixelBox, expected_box: PixelBox) {
  // if the expected box is inside the grid, return 1
  if (isInside(expected_box, grid) || isInside(grid, expected_box)) {
    return 1
  }

  // calculate the IOU, return 1 if > threshold
  let iou = calcIOU(grid, expected_box)
  // console.log({ iou, grid_score_threshold })
  return iou >= grid_score_threshold ? 1 : 0
}

function isInside(inner: PixelBox, outer: PixelBox) {
  return (
    inner.left >= outer.left &&
    inner.right <= outer.right &&
    inner.top >= outer.top &&
    inner.bottom <= outer.bottom
  )
}

function calcIOU(grid_box: PixelBox, expected_box: PixelBox): number {
  /* overlap region */
  let left = Math.max(grid_box.left, expected_box.left)
  let right = Math.min(grid_box.right, expected_box.right)
  let top = Math.max(grid_box.top, expected_box.top)
  let bottom = Math.min(grid_box.bottom, expected_box.bottom)
  let width = right - left
  let height = bottom - top
  if (width <= 0 || height <= 0) return 0
  let area_overlap = width * height

  /* union region */
  let area_grid =
    (grid_box.right - grid_box.left) * (grid_box.bottom - grid_box.top)
  let area_expected =
    (expected_box.right - expected_box.left) *
    (expected_box.bottom - expected_box.top)

  return Math.max(area_overlap / area_grid, area_overlap / area_expected)
}

function drawBox(options: {
  box: NormalizedBox
  borderColor: string
  fillColor: string
}) {
  let { box, borderColor, fillColor } = options
  let width = box.width * canvas.width
  let height = box.height * canvas.height
  let left = box.x * canvas.width - width / 2
  let top = box.y * canvas.height - height / 2
  context.strokeStyle = borderColor
  context.fillStyle = fillColor
  let size = Math.min(canvas.width, canvas.height) / 100
  context.lineWidth = size
  context.fillRect(left, top, width, height)
  context.strokeRect(left, top, width, height)
}

input.onchange = checkFile
checkFile()
