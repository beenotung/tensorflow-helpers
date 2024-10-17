import { selectImage } from '@beenotung/tslib/file'
import { loadImageModel } from './browser'
import { toOneTensor } from './tensor'

declare var selectButton: HTMLButtonElement

async function main() {
  console.time('load model')
  let model = await loadImageModel({
    url: 'saved_model/mobilenet-v3-large-100',
    cacheUrl: 'indexeddb://mobilenet-v3-large-100',
    checkForUpdates: false,
  })
  console.timeEnd('load model')
  console.log({ model })
  selectButton.disabled = false
  selectButton.onclick = async () => {
    let [file] = await selectImage({ accept: 'image/*' })
    if (!file) return
    let result = await model.imageFileToEmbedding(file)
    let tensor = toOneTensor(result)
    let data = tensor.dataSync()
    console.log({ data })
  }
}
main().catch(e => console.error(e))
