import { loadImageModel } from './browser'

async function main() {
  debugger
  let baseModel = await loadImageModel({
    url: 'saved_model/mobilenet-v3-large-100',
    cacheUrl: 'indexeddb://mobilenet-v3-large-100',
    checkForUpdates: false,
  })
  console.log({ baseModel })
}
main().catch(e => console.error(e))
