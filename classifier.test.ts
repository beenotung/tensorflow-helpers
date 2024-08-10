import { createImageClassifier } from './classifier'
import { baseModels, loadImageBaseModel } from './model'

async function main() {
  let baseModel = await loadImageBaseModel({
    spec: baseModels.mobilenet['mobilenet-v3-large-100'],
    dir: 'saved_model/base_model',
  })
  createImageClassifier({
    baseModel,
    datasetDir: 'dataset',
    // class_names,
    // hidden_layers,
  })
}
main().catch(e => console.error(e))
