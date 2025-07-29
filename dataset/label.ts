export type BoundingBox = {
  /** starts from 0 */
  class_idx: number
  /** normalized to [0,1] */
  x: number
  y: number
  width: number
  height: number
}

export type BoundingBoxWithKeypoints = BoundingBox & {
  keypoints: Keypoint[]
}

export type Keypoint = {
  x: number
  y: number
  visibility: 0 | 1
}

// Type predicates
function isKeypoint(value: unknown): value is Keypoint {
  const k = value as Keypoint
  return (
    typeof k.x === 'number' &&
    typeof k.y === 'number' &&
    (k.visibility === 0 || k.visibility === 1) &&
    isBetweenZeroAndOne(k.x) &&
    isBetweenZeroAndOne(k.y)
  )
}

export function isBoundingBox(value: unknown): value is BoundingBox {
  const b = value as BoundingBox
  return (
    typeof b.class_idx === 'number' &&
    isBetweenZeroAndOne(b.x) &&
    isBetweenZeroAndOne(b.y) &&
    isBetweenZeroAndOne(b.width) &&
    isBetweenZeroAndOne(b.height)
  )
}

export function isBoundingBoxWithKeypoints(
  value: unknown,
): value is BoundingBoxWithKeypoints {
  const b = value as BoundingBoxWithKeypoints
  return (
    isBoundingBox(b) &&
    Array.isArray(b.keypoints) &&
    b.keypoints.every(isKeypoint)
  )
}

// Validation utilities
function isBetweenZeroAndOne(value: number): boolean {
  return value >= 0 && value <= 1
}

function validateClassIndex(options: {
  class_idx: number
  n_class: number
}): void {
  const { class_idx, n_class } = options
  if (class_idx < 0 || class_idx >= n_class) {
    throw new Error(
      `Invalid class index: receive ${class_idx} but expect a range of [0,${n_class - 1}]`,
    )
  }
}

function validateBoundingBox(box: {
  x: number
  y: number
  width: number
  height: number
}): void {
  const { x, y, width, height } = box
  if (!isBetweenZeroAndOne(x) || !isBetweenZeroAndOne(y)) {
    throw new Error(
      `Invalid bounding box coordinates: x=${x}, y=${y}. Expected range [0, 1].`,
    )
  }

  if (!isBetweenZeroAndOne(width) || !isBetweenZeroAndOne(height)) {
    throw new Error(
      `Invalid bounding box size: width=${width}, height=${height}. Expected range [0, 1].`,
    )
  }
}

type BaseParseOptions = {
  line: string
  n_class: number
}

export type ParseDetectLabelOptions = BaseParseOptions

export type ParsePoseLabelOptions = BaseParseOptions & {
  n_keypoints: number
  is_visible: boolean
}

function isParsePoseLabelOptions(
  options: BaseParseOptions,
): options is ParsePoseLabelOptions {
  return 'n_keypoints' in options && 'is_visible' in options
}

// Label parsing
export function parseLabelString(
  task: 'detect' | 'pose',
  options: ParseDetectLabelOptions | ParsePoseLabelOptions,
): BoundingBox | BoundingBoxWithKeypoints {
  if (task === 'detect') {
    if (isParsePoseLabelOptions(options)) {
      throw new Error('Invalid options for detect task')
    }
    return parseDetectLabelString(options)
  } else {
    if (!isParsePoseLabelOptions(options)) {
      throw new Error('Invalid options for pose task')
    }
    return parsePoseLabelString(options)
  }
}

function parseBaseLabelString(options: BaseParseOptions) {
  const { line, n_class } = options
  const label_parts = line.trim().split(' ')

  const class_idx = +label_parts[0]
  validateClassIndex({ class_idx, n_class })

  const x = +label_parts[1]
  const y = +label_parts[2]
  const width = +label_parts[3]
  const height = +label_parts[4]
  validateBoundingBox({ x, y, width, height })

  return {
    class_idx,
    x,
    y,
    width,
    height,
    label_parts,
  }
}

function parseDetectLabelString(options: ParseDetectLabelOptions): BoundingBox {
  const { label_parts, ...bounding_box } = parseBaseLabelString(options)

  if (label_parts.length !== 5) {
    throw new Error(
      `Invalid detect (bounding box) label line: expected 5 parts but got ${label_parts.length} parts`,
    )
  }

  if (!isBoundingBox(bounding_box)) {
    throw new Error('Invalid bounding box data')
  }

  return bounding_box
}

function parsePoseLabelString(
  options: ParsePoseLabelOptions,
): BoundingBoxWithKeypoints {
  const { n_keypoints, is_visible } = options
  const { label_parts, ...bounding_box } = parseBaseLabelString(options)

  const step = is_visible ? 3 : 2
  const expected_parts = 5 + n_keypoints * step
  if (label_parts.length !== expected_parts) {
    throw new Error(
      `Invalid pose label line: expect (5 + ${n_keypoints} * ${step} = ${expected_parts}) parts but got ${label_parts.length} parts`,
    )
  }

  const keypoints: Keypoint[] = []
  for (let i = 5; i < label_parts.length; i += step) {
    const x = +label_parts[i]
    const y = +label_parts[i + 1]
    const visibility = is_visible ? (+label_parts[i + 2] as 0 | 1) : 1
    const keypoint = { x, y, visibility }

    if (!isKeypoint(keypoint)) {
      throw new Error(`Invalid keypoint at position ${i}`)
    }
    keypoints.push(keypoint)
  }

  const result = {
    ...bounding_box,
    keypoints,
  }

  if (!isBoundingBoxWithKeypoints(result)) {
    throw new Error('Invalid bounding box with keypoints data')
  }

  return result
}

// Label string generation
type BaseToLabelOptions = {
  class_idx: number
  n_class: number
  x: number
  y: number
  width: number
  height: number
}

export type DetectLabelStringOptions = BaseToLabelOptions

export type PoseLabelStringOptions = BaseToLabelOptions & {
  n_keypoints: number
  visibility: boolean
  keypoints: Keypoint[]
}

export function toLabelString(
  task: 'detect' | 'pose',
  options: DetectLabelStringOptions | PoseLabelStringOptions,
): string {
  if (task === 'detect') {
    return toDetectLabelString(options as DetectLabelStringOptions)
  } else {
    if (!isToPoseLabelStringOptions(options)) {
      throw new Error('Invalid options for pose task')
    }
    return toPoseLabelString(options)
  }
}

function isToPoseLabelStringOptions(
  options: BaseToLabelOptions,
): options is PoseLabelStringOptions {
  return (
    'n_keypoints' in options &&
    'visibility' in options &&
    'keypoints' in options
  )
}

export function toDetectLabelString(options: DetectLabelStringOptions): string {
  const { class_idx, x, y, width, height } = options
  validateClassIndex(options)
  validateBoundingBox(options)
  return `${class_idx} ${x} ${y} ${width} ${height}`
}

export function toPoseLabelString(options: PoseLabelStringOptions): string {
  const { class_idx, x, y, width, height, keypoints, n_keypoints, visibility } =
    options

  validateClassIndex(options)
  validateBoundingBox(options)

  if (n_keypoints !== keypoints.length) {
    throw new Error(
      `Number of keypoints (${keypoints.length}) does not match n_keypoints (${n_keypoints})`,
    )
  }

  if (!keypoints.every(isKeypoint)) {
    throw new Error('Invalid keypoints data')
  }

  let label = `${class_idx} ${x} ${y} ${width} ${height}`

  for (const keypoint of keypoints) {
    label += ` ${keypoint.x} ${keypoint.y}`
    if (visibility) {
      label += ` ${keypoint.visibility}`
    }
  }

  return label
}
