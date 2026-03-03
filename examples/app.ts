import {
  loadImageClassifierModel,
  loadImageModel,
  PreTrainedImageModels,
  topClassifyResult,
} from 'tensorflow-helpers/browser'

let fileInput = document.querySelector<HTMLInputElement>('#fileInput')!
let classifyButton =
  document.querySelector<HTMLButtonElement>('#classifyButton')!
let result = document.querySelector<HTMLPreElement>('#result')!
let log = document.querySelector<HTMLPreElement>('#log')!

function addLog(message: string) {
  log.textContent += '\n ' + message
  log.scrollTop = log.scrollHeight
}

fileInput.onchange = async () => {
  let file = fileInput.files?.[0]
  if (!file) return
  addLog('loaded file: ' + file.name)
}

classifyButton.onclick = async () => {
  try {
    await classifyFile()
  } catch (error) {
    addLog('error: ' + error)
  }
}

async function classifyFile() {
  let file = fileInput.files?.[0]
  if (!file) {
    addLog('no file selected')
    return
  }
  addLog('will classify file: ' + file.name)

  addLog('loading base model...')
  let baseModel = await loadImageModel({
    url: '/saved_model/base_model',
    cacheUrl: 'indexeddb://mobilenet-v3-large-100',
    checkForUpdates: false,
    cache: true,
  })
  addLog('base model loaded')

  addLog('loading classifier model...')
  let classifier = await loadImageClassifierModel({
    baseModel,
    modelUrl: '/saved_model/classifier_model',
    cacheUrl: 'indexeddb://classifier_model',
    checkForUpdates: false,
  })
  addLog('classifier model loaded')

  addLog('classifying file: ' + file.name + '...')
  let classes = await classifier.classifyImageFile(file)
  addLog('classification result: ' + JSON.stringify(classes, null, 2))

  let topClass = topClassifyResult(classes)
  addLog('top classification result: ' + JSON.stringify(topClass, null, 2))

  result.textContent = JSON.stringify(topClass, null, 2)
}
