import { mkdirSync, readdirSync } from 'fs'
import { join } from 'path'
import { cropAndResizeImageFile } from './image'

async function main() {
  const aspectRatio = 'center-crop'

  mkdirSync('resize/square', { recursive: true })
  mkdirSync('resize/tall', { recursive: true })
  mkdirSync('resize/wide', { recursive: true })
  mkdirSync('resize/down', { recursive: true })

  let filenames = readdirSync('dataset/real')
  for (let filename of filenames) {
    let srcFile = join('dataset/real', filename)

    await cropAndResizeImageFile({
      srcFile,
      destFile: join('resize/square', filename),
      width: 500,
      height: 500,
      aspectRatio,
    })
    await cropAndResizeImageFile({
      srcFile,
      destFile: join('resize/tall', filename),
      width: 160,
      height: 360,
      aspectRatio,
    })
    await cropAndResizeImageFile({
      srcFile,
      destFile: join('resize/wide', filename),
      width: 640,
      height: 180,
      aspectRatio,
    })
    await cropAndResizeImageFile({
      srcFile,
      destFile: join('resize/down', filename),
      width: 640,
      height: 360,
      aspectRatio,
    })
  }
}

main().catch(e => console.error(e))
