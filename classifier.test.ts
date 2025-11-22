import { loadImageClassifierModel, topClassifyResult } from './classifier'
import { PreTrainedImageModels, loadImageModel } from './model'

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

  let classes = await classifier.classifyImageFile('image.jpg')
  let topClass = topClassifyResult(classes)

  console.log('classes:', classes)
  console.log('top result:', topClass)
  // [print] result: { label: 'anime', confidence: 0.7991582155227661 }
}
main().catch(e => console.error(e))
