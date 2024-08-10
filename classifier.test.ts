import { createImageClassifier, loadImageClassifierModel } from './classifier'
import { PreTrainedImageModels, loadImageModel, saveModel } from './model'

async function main() {
  let baseModel = await loadImageModel({
    spec: PreTrainedImageModels.mobilenet['mobilenet-v3-large-100'],
    dir: 'saved_model/base_model',
  })
  let classifier = await loadImageClassifierModel({
    baseModel,
    classifierModelDir: 'saved_model/classifier_model',
    datasetDir: 'dataset',
    // class_names,
    // hidden_layers,
  })
  let history = await classifier.trainAsync({
    epochs: 5,
    batchSize: 32,
  })
  // console.log('history:', history)

  await classifier.save()
}
main().catch(e => console.error(e))
