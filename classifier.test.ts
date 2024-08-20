import {
  loadImageClassifierModel,
  topClassifyResult as topClassificationResult,
} from './classifier'
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
    hiddenLayers: [128],
    datasetDir: 'dataset',
    // classNames: ['anime', 'real'], // auto scan from datasetDir
  })
  let { x, y } = await classifier.loadDatasetFromDirectoryAsync()
  let history = await classifier.trainAsync({
    x,
    y,
    epochs: 5,
    batchSize: 32,
  })
  // console.log('history:', history)

  // reuse the encoded image dataset
  await classifier.trainAsync({
    x,
    y,
    epochs: 5,
    batchSize: 32,
  })

  await classifier.save()

  let classes = await classifier.classifyAsync('image.jpg')
  let topClass = topClassificationResult(classes)

  console.log('result:', topClass)
  // [print] result: { label: 'anime', confidence: 0.7991582155227661 }
}
main().catch(e => console.error(e))
