import { readdirSync } from 'fs'
import { loadImageClassifierModel, topClassifyResult } from './classifier'
import { PreTrainedImageModels, loadImageModel } from './model'
import { join } from 'path'

async function main() {
  let baseModel = await loadImageModel({
    spec: PreTrainedImageModels.mobilenet['mobilenet-v3-large-100'],
    dir: 'saved_model/base_model',
  })
  console.log('embedding features:', baseModel.spec.features)
  // [print] embedding features: 1280

  let classifier = await loadImageClassifierModel({
    baseModel,
    modelDir: 'saved_model/classifier_model',
    // hiddenLayers: [128],
    hiddenLayers: [baseModel.spec.features],
    datasetDir: 'dataset',
    // classNames: ['others', 'anime', 'real'], // auto scan from datasetDir
  })
  let { x, y } = await classifier.loadDatasetFromDirectory()
  let history = await classifier.train({
    x,
    y,
    epochs: 5,
    batchSize: 32,
  })
  // console.log('history:', history)

  await classifier.save()

  let dir = 'images'
  for (let filename of readdirSync(dir)) {
    console.log('-'.repeat(32))
    let file = join(dir, filename)
    let classes = await classifier.classifyImageFile(file)
    let topClass = topClassifyResult(classes)
    console.log('file:', file)
    console.log('classes:', classes)
    console.log('top result:', topClass)
    // [print] result: { label: 'anime', confidence: 0.7991582155227661 }
    console.log('-'.repeat(32))
  }
}
main().catch(e => console.error(e))
