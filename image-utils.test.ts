import { cropAndResizeImageFile } from './image'

async function main() {
  await cropAndResizeImageFile({
    srcFile: 'samples/wide.jpg',
    destFile: 'samples/wide-square.jpg',
    width: 224,
    height: 224,
    aspectRatio: 'center-crop',
  })

  await cropAndResizeImageFile({
    srcFile: 'samples/tall.jpg',
    destFile: 'samples/tall-square.jpg',
    width: 224,
    height: 224,
    aspectRatio: 'center-crop',
  })
}
main().catch(e => console.error(e))
