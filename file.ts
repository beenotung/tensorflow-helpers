import { createHash } from 'crypto'
import { getDirFilenames } from '@beenotung/tslib/fs'
import { readFile, rename, writeFile } from 'fs/promises'
import { basename, dirname, extname, join } from 'path'

/**
 * @description
 * - rename filename to content hash + extname;
 * - return list of (renamed) filenames
 */
export async function scanDir(dir: string) {
  let filenames = await getDirFilenames(dir)
  for (let i = 0; i < filenames.length; i++) {
    let filename = filenames[i]
    if (!isContentHash(filename)) {
      filenames[i] = await renameFileByContentHash(join(dir, filename))
    }
  }
  return filenames
}

export function isContentHash(file_or_filename: string): boolean {
  let filename = basename(file_or_filename)
  let ext = extname(filename)
  let name = ext.length == 0 ? filename : filename.slice(0, -ext.length)
  return name.length * 4 == 256 && Buffer.from(name, 'hex').length * 8 == 256
}

export async function saveFile(args: {
  dir: string
  content: Buffer
  mimeType: string
}) {
  let ext = args.mimeType.split('/').pop()!.split(';')[0]
  let filename = hashContent(args.content) + '.' + ext
  let file = join(args.dir, filename)
  await writeFile(file, args.content)
}

export function hashContent(content: Buffer, encoding: BufferEncoding = 'hex') {
  let hash = createHash('sha256')
  hash.write(content)
  return hash.digest().toString(encoding)
}

/** @returns new filename with content hash and extname */
export async function renameFileByContentHash(file: string) {
  let dir = dirname(file)
  let filename = basename(file)
  let ext = extname(filename)
  let content = await readFile(file)
  let hash = hashContent(content)
  let newFilename = hash + ext
  let newFile = join(dir, newFilename)
  await rename(file, newFile)
  return newFile
}
