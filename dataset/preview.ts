import { mkdirSync } from 'fs'
import { extname, basename, join } from 'path'
import { DetectYamlOptions, PoseYamlOptions } from './yaml'
import {
  BoundingBox,
  BoundingBoxWithKeypoints,
  isBoundingBoxWithKeypoints,
} from './label'
import { drawBox } from 'yolo-helpers/dist/canvas'
import { createCanvas, loadImage } from 'canvas'
import { writeFile } from 'fs/promises'
import { DatasetOptions, BoundingBoxGroups, ImageLabelDict } from './fs'

// export type PreviewPathImagePair = {
//   [preview_path: string]: string
// }

type PreviewBase64Pair = {
  preview_path_arr: string[]
  base64_arr: string[]
}

type PreviewBase64PairByGroup = {
  train: PreviewBase64Pair
  test: PreviewBase64Pair
  val: PreviewBase64Pair
}

export async function getPreviewPathsAndBase64(
  options: DatasetOptions,
): Promise<PreviewBase64PairByGroup> {
  const { dir, task, metadata, bounding_box_groups } = options

  const group_types: Array<keyof PreviewBase64PairByGroup> = [
    'train',
    'test',
    'val',
  ]

  const preview_path_base64_groups: Partial<PreviewBase64PairByGroup> = {}

  for (const group_type of group_types) {
    preview_path_base64_groups[group_type] =
      await getPreviewBase64ArrForOneGroup(task, {
        dir,
        group_type,
        bounding_box_dict: bounding_box_groups[group_type],
        metadata,
      })
  }

  console.log('Finish getting preview paths and images')
  return preview_path_base64_groups as PreviewBase64PairByGroup
}

export async function getPreviewBase64ArrForOneGroup<
  T extends DetectYamlOptions | PoseYamlOptions,
>(
  task: 'detect' | 'pose',
  options: {
    dir: string
    group_type: string
    bounding_box_dict: ImageLabelDict<BoundingBox | BoundingBoxWithKeypoints>
    metadata: T
  },
): Promise<PreviewBase64Pair> {
  const { dir, group_type, bounding_box_dict, metadata } = options
  const preview_dir_path = join(dir, group_type, 'previews')

  const preview_path_arr: string[] = []
  const base64_arr: string[] = []

  for (const image_filename in bounding_box_dict) {
    const image_full_path = join(dir, group_type, 'images', image_filename)
    const base64 = await getOnePreviewBase64FromBoxes(task, {
      image_full_path,
      metadata,
      boxes: bounding_box_dict[image_filename],
    })
    const preview_file_path = join(preview_dir_path, basename(image_full_path))

    preview_path_arr.push(preview_file_path)
    base64_arr.push(base64)
  }
  return { preview_path_arr, base64_arr }
}

export async function getOnePreviewBase64FromBoxes<
  T extends DetectYamlOptions | PoseYamlOptions,
>(
  task: 'detect' | 'pose',
  options: {
    image_full_path: string
    metadata: T
    boxes: (BoundingBox | BoundingBoxWithKeypoints)[]
  },
): Promise<string> {
  const { image_full_path, metadata, boxes } = options

  const image = await loadImage(image_full_path)
  const canvas = createCanvas(image.width, image.height)
  const context = canvas.getContext('2d')!
  context.drawImage(image, 0, 0, image.width, image.height)

  for (const label of boxes) {
    const class_label =
      metadata.class_names?.[label.class_idx] || `class_${label.class_idx}`

    drawBox({
      context: context as any,
      x: label.x * image.width,
      y: label.y * image.height,
      width: label.width * image.width,
      height: label.height * image.height,
      label: { text: class_label },
    })

    if (task === 'pose' && isBoundingBoxWithKeypoints(label)) {
      label.keypoints.forEach((keypoint, index) => {
        const keypoint_label = [
          (metadata as PoseYamlOptions).keypoint_names?.[index] ||
            index.toString(),
          (metadata as PoseYamlOptions).visibility
            ? `(${keypoint.visibility})`
            : '',
        ]
          .filter(Boolean)
          .join(' ')

        drawBox({
          context: context as any,
          x: keypoint.x * image.width,
          y: keypoint.y * image.height,
          width: 3,
          height: 3,
          borderColor: '#39FF14',
          label: {
            text: keypoint_label,
            fontColor: '#14efff',
            backgroundColor: '#000005',
          },
        })
      })
    }
  }
  const base64 = canvas.toDataURL().split(',')[1]
  return base64
}
