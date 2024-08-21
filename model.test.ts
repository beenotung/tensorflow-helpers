import { PreTrainedImageModels, loadImageModel } from './model'
import { writeFile } from 'fs/promises'

async function main() {
  let res = await fetch('https://picsum.photos/id/237/536/354')
  let buffer = await res.arrayBuffer()
  let file = 'image.jpg'
  await writeFile(file, Buffer.from(buffer))

  let baseModel = await loadImageModel({
    spec: PreTrainedImageModels.mobilenet['mobilenet-v3-large-100'],
    dir: 'saved_model/base_model',
  })

  let embedding = await baseModel.imageFileToEmbedding(file)

  console.log(embedding)
}
main().catch(e => console.error(e))
