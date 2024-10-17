import { selectImage } from '@beenotung/tslib/file'
import {
  loadImageClassifierModel,
  loadImageModel,
  toOneTensor,
} from './browser'

declare var selectButton: HTMLButtonElement

async function main() {
  console.time('load base model')
  let baseModel = await loadImageModel({
    url: 'saved_model/mobilenet-v3-large-100',
    cacheUrl: 'indexeddb://mobilenet-v3-large-100',
    checkForUpdates: false,
  })
  console.timeEnd('load base model')
  console.log({ baseModel })

  let classifier = await loadImageClassifierModel({
    baseModel,
    classNames: ['happy', 'sad', 'normal', 'others'],
    modelUrl: 'saved_model/emotion-classifier',
    cacheUrl: 'indexeddb://emotion-classifier',
  })

  console.log({ classifier })

  selectButton.disabled = false
  selectButton.onclick = async () => {
    let [file] = await selectImage({ accept: 'image/*' })
    if (!file) return

    console.time('get image embedding')
    let embeddingTensor = await baseModel.imageFileToEmbedding(file)
    let embeddingFeatures = toOneTensor(embeddingTensor).dataSync()
    console.timeEnd('get image embedding')
    console.log({ embeddingFeatures })

    console.time('classify image embedding')
    let result = await classifier.classifyImageEmbedding(embeddingTensor)
    console.timeEnd('classify image embedding')
    console.log({ result })

    console.time('classify image file')
    result = await classifier.classifyImageFile(file)
    console.timeEnd('classify image file')
    console.log({ result })
  }
}
main().catch(e => console.error(e))
