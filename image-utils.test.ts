import { cropAndResizeImageFile } from './image'

async function main() {
  await cropAndResizeImageFile({
    srcFile: 'dataset/wide.jpg',
    destFile: 'dataset/wide-square.jpg',
    width: 224,
    height: 224,
    aspectRatio: 'center-crop',
  })

  await cropAndResizeImageFile({
    srcFile: 'dataset/tall.jpg',
    destFile: 'dataset/tall-square.jpg',
    width: 224,
    height: 224,
    aspectRatio: 'center-crop',
  })
}
main().catch(e => console.error(e))
