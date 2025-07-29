import { readFileSync, existsSync, mkdirSync, writeFileSync } from 'fs'
import { extname, basename, join } from 'path'
import {
  DetectYamlOptions,
  PoseYamlOptions,
  parseDataYaml,
  toDataYamlString,
} from './dataset/yaml'
import {
  BoundingBox,
  BoundingBoxWithKeypoints,
  parseLabelString,
  ParseDetectLabelOptions,
  ParsePoseLabelOptions,
  toPoseLabelString,
  toDetectLabelString,
} from './dataset/label'
import { getDirFilenamesSync } from '@beenotung/tslib/fs'
import { writeFile } from 'fs/promises'
import { extract_lines } from '@beenotung/tslib/string'
import { zipArray } from '@beenotung/tslib'
import {
  getPreviewPathsAndBase64,
  getPreviewBase64ArrForOneGroup,
  getOnePreviewBase64FromBoxes as getOnePreviewBase64FromBoxes,
} from './dataset/preview'

export function validateDatasetDir(
  dataset_dir: string,
  yaml_filename: string,
): void {
  const requiredPaths = [
    join(dataset_dir, yaml_filename),
    ...['train', 'test', 'val'].flatMap(group =>
      ['images', 'labels'].map(type => join(dataset_dir, group, type)),
    ),
  ]
  requiredPaths.forEach(path => {
    if (!existsSync(path)) throw new Error(`${path} does not exist`)
  })
}

const image_extensions = ['.jpg', '.jpeg', '.png']

export function getImagePaths(dir: string, group: string): string[] {
  return getDirFilenamesSync(join(dir, group, 'images')).filter(file => {
    if (!image_extensions.includes(extname(file))) {
      throw new Error(`Error: Unsupported image type (Given ${extname(file)})`)
    }
    return true
  })
}

export function getLabelPaths(
  dir: string,
  group: string,
  image_paths: string[],
): string[] {
  return getDirFilenamesSync(join(dir, group, 'labels')).filter(label_file => {
    const label_name = basename(label_file, extname(label_file))
    const match = image_paths.some(
      image => basename(image, extname(image)) === label_name,
    )
    if (!match) {
      throw new Error(`Error: Label missing for ${label_file}`)
    }
    return true
  })
}

// ==================== Label Parsing ====================
export function parseLabelData<T extends 'detect' | 'pose'>(
  task: T,
  options: T extends 'pose'
    ? Omit<ParsePoseLabelOptions, 'line'>
    : Omit<ParseDetectLabelOptions, 'line'>,
  content: string,
): T extends 'pose' ? BoundingBoxWithKeypoints[] : BoundingBox[] {
  const lines = extract_lines(content)
  const result: (BoundingBox | BoundingBoxWithKeypoints)[] = []

  for (const line of lines) {
    if (!line.trim()) continue

    const parseOptions = { ...options, line }
    const box = parseLabelString(task, parseOptions)
    result.push(box)
  }

  return result as T extends 'pose' ? BoundingBoxWithKeypoints[] : BoundingBox[]
}

// ==================== Metadata Handling ====================
export function getMetaData(
  task: 'detect' | 'pose',
  options: {
    dir: string
    yaml_filename: string
  },
): DetectYamlOptions | PoseYamlOptions {
  const path = join(options.dir, options.yaml_filename)
  const yaml_content = readFileSync(path).toString()
  return parseDataYaml(task, yaml_content)
}

// ==================== Image/Label Validation ====================
export function getValidatedImageAndLabels(options: {
  image_dir: string
  label_dir: string
  image_paths: string[]
  label_paths: string[]
  missing_labels: 'error' | 'warn' | 'ignore'
}) {
  const { image_dir, label_dir, image_paths, label_paths, missing_labels } =
    options

  const validated_image_paths = image_paths.filter(image_path => {
    const image_full_path = join(image_dir, image_path)
    const label_path = basename(image_path, extname(image_path)) + '.txt'
    const label_full_path = join(label_dir, label_path)

    if (!existsSync(label_full_path)) {
      handleMissingLabel(
        missing_labels,
        `Label file not found for image: ${image_full_path}`,
      )
      return false
    }
    return true
  })

  const validated_label_paths = label_paths.filter(label_path => {
    const label_full_path = join(label_dir, label_path)
    const label_filename = basename(label_path, extname(label_path))
    const hasMatchingImage = image_extensions.some(ext =>
      validated_image_paths.includes(label_filename + ext),
    )

    if (!hasMatchingImage) {
      handleMissingLabel(
        missing_labels,
        `Image file not found for label: ${label_full_path}`,
      )
      return false
    }
    return true
  })

  if (validated_image_paths.length !== validated_label_paths.length) {
    throw new Error(
      `Mismatch between number of images and labels: ${validated_image_paths.length} images, ${validated_label_paths.length} labels`,
    )
  }

  return {
    validated_image_paths: validated_image_paths.sort(),
    validated_label_paths: validated_label_paths.sort(),
  }
}

function handleMissingLabel(
  mode: 'error' | 'warn' | 'ignore',
  message: string,
): void {
  if (mode === 'error') throw new Error(message)
  if (mode === 'warn') console.warn(`Warning: ${message}`)
}

// ==================== Get Bounding Boxes ====================
export async function getBoundingBoxesDict<
  T extends DetectYamlOptions | PoseYamlOptions,
>(
  task: 'detect' | 'pose',
  options: {
    dir: string
    group_type: string
    image_paths: string[]
    label_paths: string[]
    metadata: T
    missing_labels?: 'error' | 'warn' | 'ignore'
  },
): Promise<ImageLabelDict> {
  const { dir, group_type, metadata, missing_labels = 'warn' } = options

  const { validated_image_paths, validated_label_paths } =
    getValidatedImageAndLabels({
      ...options,
      missing_labels,
      image_dir: join(dir, group_type, 'images'),
      label_dir: join(dir, group_type, 'labels'),
    })

  const bounding_box_dict: ImageLabelDict = {}

  for (let i = 0; i < validated_image_paths.length; i++) {
    const image_full_path = join(
      dir,
      group_type,
      'images',
      validated_image_paths[i],
    )
    const label_full_path = join(
      dir,
      group_type,
      'labels',
      validated_label_paths[i],
    )
    const filename = basename(image_full_path)

    const boxes = await getBoundingBoxesOfOneLabel(task, {
      image_full_path,
      label_full_path,
      metadata,
    })
    bounding_box_dict[filename] = boxes
  }

  return bounding_box_dict
}

export async function getBoundingBoxesOfOneLabel<
  T extends DetectYamlOptions | PoseYamlOptions,
>(
  task: 'detect' | 'pose',
  options: {
    image_full_path: string
    label_full_path: string
    metadata: T
  },
): Promise<(BoundingBox | BoundingBoxWithKeypoints)[]> {
  const { label_full_path, metadata } = options

  const content = readFileSync(label_full_path, 'utf-8')
  const label_data_boxes = parseLabelData(
    task,
    task === 'pose'
      ? {
          n_class: metadata.n_class,
          is_visible: (metadata as PoseYamlOptions).visibility,
          n_keypoints: (metadata as PoseYamlOptions).n_keypoints,
        }
      : { n_class: metadata.n_class },
    content,
  )

  return label_data_boxes
}

// ==================== Generate Preview Image ====================
export async function createPreviewImages(options: DatasetOptions) {
  const { dir } = options
  const preview_path_base64_pair_groups =
    await getPreviewPathsAndBase64(options)
  const group_types = ['train', 'test', 'val'] satisfies Array<
    keyof BoundingBoxGroups
  >
  for (const group_type of group_types) {
    mkdirSync(join(dir, group_type, 'previews'), { recursive: true })
    for (const [preview_path, base64] of zipArray(
      preview_path_base64_pair_groups[group_type].preview_path_arr,
      preview_path_base64_pair_groups[group_type].base64_arr,
    )) {
      await writeFile(preview_path, base64, 'base64')
    }
  }
  console.log('Finish creating preview images')
}

export async function createPreviewImagesForOneGroup<
  T extends DetectYamlOptions | PoseYamlOptions,
>(
  task: 'detect' | 'pose',
  options: {
    dir: string
    group_type: string
    bounding_box_dict: ImageLabelDict<BoundingBox | BoundingBoxWithKeypoints>
    metadata: T
  },
): Promise<void> {
  const { dir, group_type, bounding_box_dict, metadata } = options
  const preview_dir_path = join(dir, group_type, 'previews')
  mkdirSync(preview_dir_path, { recursive: true }) //refactor

  const result = await getPreviewBase64ArrForOneGroup(task, {
    dir: preview_dir_path,
    group_type,
    metadata,
    bounding_box_dict,
  })
  for (const [preview_path, base64] of zipArray(
    result.preview_path_arr,
    result.base64_arr,
  )) {
    await writeFile(preview_path, base64, 'base64')
  }
  console.log('Finish creating preview images')
}

export async function createOnePreviewImage<
  T extends DetectYamlOptions | PoseYamlOptions,
>(
  task: 'detect' | 'pose',
  options: {
    image_full_path: string
    preview_dir_path: string
    metadata: T
    boxes: (BoundingBox | BoundingBoxWithKeypoints)[]
  },
): Promise<void> {
  const { image_full_path, preview_dir_path } = options
  const base64 = await getOnePreviewBase64FromBoxes(task, options)

  const image_name = basename(image_full_path)
  const preview_path = join(preview_dir_path, image_name)
  await writeFile(preview_path, base64, 'base64')
}

// ==================== File Utilities ====================
export function saveYAMLFile(
  file: string,
  content: string,
): 'no change' | 'saved' {
  const newContent = content.trim() + '\n'

  if (existsSync(file)) {
    const oldContent = readFileSync(file).toString()
    if (newContent === oldContent) return 'no change'
  }

  writeFileSync(file, newContent)
  return 'saved'
}

export function saveJsonFile(
  save_file_path: string,
  data: any,
  options?: { verbose?: boolean },
) {
  let str = '{'
  for (let test of data) {
    str += JSON.stringify({ [test[0]]: test[1] }).slice(1, -1)
    str += ',\n'
  }
  str = str.slice(0, -2)
  str += '\n}'

  writeFileSync(save_file_path, str)
  if (options?.verbose) {
    console.log('saved to file:', save_file_path)
  }
}

// ==================== Dataset Options Types ====================
export type DatasetOptions = DetectDatasetOptions | PoseDatasetOptions

type BaseDatasetOptions = {
  dir: string
}
export type DetectDatasetOptions = BaseDatasetOptions & {
  task: 'detect'
  metadata: DetectYamlOptions
  bounding_box_groups: BoundingBoxGroups<BoundingBox>
}
export type PoseDatasetOptions = BaseDatasetOptions & {
  task: 'pose'
  metadata: PoseYamlOptions
  bounding_box_groups: BoundingBoxGroups<BoundingBoxWithKeypoints>
}

export type BoundingBoxGroups<Box = BoundingBox> = {
  train: ImageLabelDict<Box>
  test: ImageLabelDict<Box>
  val: ImageLabelDict<Box>
}

export type ImageLabelDict<Box = BoundingBox> = {
  [image_filename: string]: Box[]
}

// ==================== Dataset Import (From specified directory) ====================
// returns metadata & train/test/val bounding boxes dicts
export async function importDataset(options: {
  dir: string
  task: 'detect'
  yaml_filename?: string
}): Promise<Omit<DetectDatasetOptions, 'dir' | 'task'>>
export async function importDataset(options: {
  dir: string
  task: 'pose'
  yaml_filename?: string
}): Promise<Omit<PoseDatasetOptions, 'dir' | 'task'>>
export async function importDataset(options: {
  dir: string
  task: 'detect' | 'pose'
  yaml_filename?: string
}): Promise<Omit<DatasetOptions, 'dir' | 'task'>> {
  const { dir, task } = options
  const yaml_filename = options.yaml_filename ?? 'data.yaml'
  validateDatasetDir(dir, yaml_filename)

  const path = join(dir, yaml_filename)
  const yaml_content = readFileSync(path, 'utf-8')
  const metadata = parseDataYaml(task, yaml_content)

  const bounding_box_dicts = []

  for (const group_type of ['train', 'test', 'val']) {
    const image_paths = getImagePaths(dir, group_type)
    const label_paths = getLabelPaths(dir, group_type, image_paths)
    const bounding_box_dict = await getBoundingBoxesDict(task, {
      dir,
      group_type,
      image_paths,
      label_paths,
      metadata: metadata,
    })
    bounding_box_dicts.push(bounding_box_dict)
  }

  const [
    bounding_box_dict_train,
    bounding_box_dict_test,
    bounding_box_dict_val,
  ] = bounding_box_dicts

  const bounding_box_groups: BoundingBoxGroups = {
    train: bounding_box_dict_train,
    test: bounding_box_dict_test,
    val: bounding_box_dict_val,
  }

  console.log('Imported dataset')
  return {
    metadata,
    bounding_box_groups,
  }
}

// ==================== Dataset Export (To specified directory) ====================
export async function exportDataset(options: DatasetOptions): Promise<void> {
  const { dir, task, metadata } = options
  createExportDatasetDirs(dir)

  const yaml_path = join(dir, 'data.yaml')
  const yaml_str = toDataYamlString(task, metadata)
  await writeFile(yaml_path, yaml_str)

  const groups = ['train', 'test', 'val'] satisfies Array<
    keyof BoundingBoxGroups
  >

  for (const group_type of groups) {
    const task = options.task
    switch (options.task) {
      case 'detect': {
        const { metadata } = options
        await saveLabelFiles(
          options.dir,
          group_type,
          options.bounding_box_groups[group_type],
          box => toDetectLabelString({ ...box, ...metadata }),
        )
        break
      }
      case 'pose': {
        const { metadata } = options
        await saveLabelFiles(
          options.dir,
          group_type,
          options.bounding_box_groups[group_type],
          box => toPoseLabelString({ ...box, ...metadata }),
        )
        break
      }
      default: {
        options satisfies never
        throw new Error(`unknown task type "${task}"`)
      }
    }
  }

  console.log('Exported dataset')
}

function createExportDatasetDirs(export_dir_path: string): void {
  const group_types = ['train', 'test', 'val']
  const data_types = ['images', 'labels']

  for (const group_type of group_types) {
    for (const data_type of data_types) {
      const dir_path = join(export_dir_path, group_type, data_type)
      if (!existsSync(dir_path)) {
        mkdirSync(dir_path, { recursive: true })
      }
    }
  }
}

export function toLabelFilename(image_filename: string) {
  const image_extname = extname(image_filename)
  return basename(image_filename, image_extname) + '.txt'
}

export async function saveLabelFile(file: string, lines: string[]) {
  const content = lines.filter(line => line.length > 0).join('\n') + '\n'
  await writeFile(file, content)
}

async function saveLabelFiles<Box>(
  dataset_dir: string,
  group_type: string,
  dict: ImageLabelDict<Box>,
  toLabelString: (box: Box) => string,
) {
  for (const image_filename in dict) {
    const boxes = dict[image_filename]
    const lines = boxes.map(toLabelString)
    const label_file = join(
      dataset_dir,
      group_type,
      'labels',
      toLabelFilename(image_filename),
    )
    await saveLabelFile(label_file, lines)
  }
}

async function test() {
  const dir = 'cat-pose-dataset_kpt6'
  const task = 'pose'
  // const task = 'detect'

  const result = await importDataset({ dir, task })
  //optional create preview images
  await createPreviewImages({ task, dir, ...result })

  const export_dir = 'dataset2'
  await exportDataset({
    dir: export_dir,
    task,
    ...result,
  })
}
test()
