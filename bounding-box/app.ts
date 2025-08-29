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

  let classifier1 = await loadLayersModel({
    url: './saved_models/c1-model',
  })

  let classifier2 = await loadLayersModel({
    url: './saved_models/c2-model',
  })
  

  return { baseModel, classifier1, classifier2 }
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

  let { baseModel, classifier1, classifier2 } = await modelsPromise

  await predict()

  async function predict() {
    let embedding = await getImageFeatures({
      tf,
      imageModel: baseModel,
      image: dataUrl,
    })
    let spatialFeatures = embedding.spatialFeatures
    let result2D = classifier1.predict(spatialFeatures) as tf.Tensor
    // 1x7x7x2
    let [matrix] = (await result2D.array()) as number[][][][]

    let targets = [] as [x: number, y: number][]
    const target_threshold = 0.3

    for (let y = 0; y < matrix.length; y++) {
      for (let x = 0; x < matrix[y].length; x++) {
        let [background, object] = matrix[y][x]
        if (object >= target_threshold) {
          targets.push([x, y])
        }
      }
    }
    if (targets.length === 0) {
      targets.push([0,0])
      targets.push([COL-1, ROW-1])
    }
    let target_x = targets.map((t) => t[0])
    let target_y = targets.map((t) => t[1])

    let predicted_box_1: NormalizedBox = {
      x: (Math.min(...target_x) + Math.max(...target_x) + 1) / (2 * COL),
      y: (Math.min(...target_y) + Math.max(...target_y) + 1) / (2 * ROW),
      width: (Math.max(...target_x) - Math.min(...target_x) + 1) / COL,
      height: (Math.max(...target_y) - Math.min(...target_y) + 1) / ROW,
    }

    let pixel_predicted_box: PixelBox = {
      left: Math.min(...target_x) * 32 ,
      top: Math.min(...target_y) * 32,
      right: (Math.max(...target_x) + 1) * 32 ,
      bottom: (Math.max(...target_y) + 1) * 32,
    }
    let predicted_box_width = pixel_predicted_box.right - pixel_predicted_box.left
    let predicted_box_height = pixel_predicted_box.bottom - pixel_predicted_box.top

    //rectangle to square option
    // let left =  Math.min(...target_x) * 32 
    // let top = Math.min(...target_y) * 32
    // let right = (Math.max(...target_x) + 1) * 32 
    // let bottom = (Math.max(...target_y) + 1) * 32
    // let predicted_box_width = right - left
    // let predicted_box_height = bottom - top
    // let width_height_diff = predicted_box_width - predicted_box_height
    // let pixel_predicted_box: PixelBox

    // if (width_height_diff > 0) {
    //   //width > height, expand height
    //   predicted_box_height = width_height_diff
    //   top -= width_height_diff / 2
    //   bottom += width_height_diff / 2
    //   if (top < 0) {
    //     bottom -= top
    //     top = 0
    //   }
    //   if (bottom > 224) {
    //     top -= (bottom - 224)
    //     bottom = 224
    //   }
    // }
    // else if (width_height_diff < 0) {
    //   //height > width, expand width
    //   predicted_box_width -= width_height_diff
    //   left += width_height_diff / 2
    //   right -= width_height_diff / 2
    //   if (left < 0) {
    //     right -= left
    //     left = 0
    //   }
    //   if (right > 224) {
    //     left -= (right - 224)
    //     right = 224
    //   }
    // }
    // pixel_predicted_box = {
    //   left,
    //   top,
    //   right,
    //   bottom,
    // }

    const canvas_temp = document.createElement('canvas');

    const originalWidth = image.naturalWidth;
    const originalHeight = image.naturalHeight;

    // const ctx_temp = canvas_temp.getContext('2d')!;
    const ctx_temp = canvas_temp.getContext('2d')!;
    ctx_temp.drawImage(image, 
      Math.round(pixel_predicted_box.left * originalWidth / image.width), 
      Math.round(pixel_predicted_box.top * originalHeight / image.height), 
      Math.round(predicted_box_width * originalWidth / image.width), 
      Math.round(predicted_box_height * originalHeight / image.height),
      0, 0, canvas.width, canvas.height);
    
    let embedding2 = await getImageFeatures({
      tf,
      imageModel: baseModel,
      image: canvas_temp!,
    })

    spatialFeatures = embedding2.spatialFeatures;
    let result2D_ = classifier2.predict(spatialFeatures) as tf.Tensor
    [matrix] = (await result2D_.array()) as number[][][][]
    targets = [] 
    let small_box_width = predicted_box_width / COL
    let small_box_height = predicted_box_height / ROW

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
          fillColor: `rgba(255,0,0,0)`,
        })
        drawBox({
          box: {
            x: (pixel_predicted_box.left + (x + 0.5) * small_box_width)/ 224,
            y: (pixel_predicted_box.top + (y + 0.5) * small_box_height) / 224,
            width: small_box_width / image.width,
            height: small_box_height/ image.height,
          },
          borderColor: 'rgba(255, 255, 0, 0)',
          fillColor: `rgba(255,0,0,${opacity})`,
        })
      }
    }
    if (targets.length === 0) {
      //if no target, use center grid
      targets.push([0,0])
      targets.push([COL-1, ROW-1])
    }
    target_x = targets.map((t) => t[0])
    target_y = targets.map((t) => t[1])

    //in cropped
    let predicted_box_: NormalizedBox = {
      x: (Math.min(...target_x) + Math.max(...target_x) + 1) / (2 * COL),
      y: (Math.min(...target_y) + Math.max(...target_y) + 1) / (2 * ROW),
      width: (Math.max(...target_x) - Math.min(...target_x) + 1) / COL,
      height: (Math.max(...target_y) - Math.min(...target_y) + 1) / ROW,
    }

    //as whole img
    let predicted_box : NormalizedBox = {
      x: (pixel_predicted_box.left + predicted_box_.x * predicted_box_width )/ image.width,
      y: (pixel_predicted_box.top + predicted_box_.y * predicted_box_height )/ image.height,
      width: predicted_box_.width * predicted_box_width / image.width,
      height: predicted_box_.height * predicted_box_height / image.height ,
    }
    console.log({ predicted_box })

    /* draw bounding box */
    drawBox({
      box: expected_box,
      borderColor: '#00ff00',
      fillColor: 'transparent',
    })
    drawBox({
      box: predicted_box_1,
      borderColor: '#ff0000',
      fillColor: 'transparent',
    })
    drawBox({
      box: predicted_box,
      borderColor: '#0077ffff',
      fillColor: 'transparent',
    })
  }
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
