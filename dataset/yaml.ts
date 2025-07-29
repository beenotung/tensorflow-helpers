function toString(data: unknown): string {
  return Array.isArray(data)
    ? '[' + data.map(toString).join(', ') + ']'
    : JSON.stringify(data)
}

function removeComment(line: string) {
  return line.split('#')[0]
}

function replaceStringQuote(line: string) {
  return line.replaceAll("'", '"')
}

function isDigit(char: string) {
  return '0' <= char && char <= '9'
}

function hasValue(lines: string[], name: string) {
  let pattern = name.toLowerCase() + ':'
  let index = lines.findIndex(line => line.toLowerCase().startsWith(pattern))
  return index != -1
}

function findValue(lines: string[], name: string) {
  let pattern = name.toLowerCase() + ':'
  let index = lines.findIndex(line => line.toLowerCase().startsWith(pattern))
  let line = lines[index]
  if (!line) return
  let rest = removeComment(line.slice(pattern.length)).trim()
  if (rest.startsWith('[')) {
    // parse inline array
    return parseValue(rest)
  }
  if (!rest) {
    // parse multiline array or object
    return Array.from(parseMultilineArray(lines, index))
  }
  // number or string
  return parseValue(rest)
}

function isString(value: unknown): value is string {
  return typeof value === 'string'
}

function isNumber(value: unknown): value is number {
  return typeof value === 'number'
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(isString)
}

function isNumberArray(value: unknown): value is number[] {
  return Array.isArray(value) && value.every(isNumber)
}

function findString(lines: string[], name: string): string {
  let value = findValue(lines, name)
  if (!isString(value)) {
    throw new TypeError(`expect ${name} to be string, but got: ${typeof value}`)
  }
  return value
}

function findNumber(lines: string[], name: string): number {
  let value = findValue(lines, name)
  if (!isNumber(value)) {
    throw new TypeError(`expect ${name} to be number, but got: ${typeof value}`)
  }
  return value
}

function findStringArray(lines: string[], name: string): string[] {
  let value = findValue(lines, name)
  if (!isStringArray(value)) {
    throw new TypeError(
      `expect ${name} to be string array, but got: ${Array.isArray(value) ? 'array of other type' : typeof value}`,
    )
  }
  return value
}

function findNumberArray(lines: string[], name: string): number[] {
  let value = findValue(lines, name)
  if (!isNumberArray(value)) {
    throw new TypeError(
      `expect ${name} to be number array, but got: ${Array.isArray(value) ? 'array of other type' : typeof value}`,
    )
  }
  return value
}

function parseValue(value: string): unknown {
  value = removeComment(value)
  value = value.trim()
  value = replaceStringQuote(value)
  // inline array
  if (value.startsWith('[')) {
    return JSON.parse(value)
  }
  // string with quote
  if (value.startsWith('"')) {
    return JSON.parse(value)
  }
  // number
  if (isDigit(value[0])) {
    return +value
  }
  // string
  return value
}

function* parseMultilineArray(lines: string[], index: number) {
  for (let i = index + 1; i < lines.length; i++) {
    // e.g. "- value" or "0: value"
    let line = lines[i].trim()

    // test "- value"
    if (line.startsWith('- ')) {
      yield parseValue(line.slice(2).trim())
      continue
    }

    // test "0: value"
    let [key, value] = line.split(':')
    if (+key) {
      yield parseValue(value)
      continue
    }

    break
  }
}

type BaseYamlOptions = {
  train_path: string
  val_path: string
  test_path: string
  n_class: number
  class_names?: string[]
}

export type DetectYamlOptions = BaseYamlOptions

export type PoseYamlOptions = BaseYamlOptions & {
  keypoint_names?: string[]
  n_keypoints: number
  visibility: boolean
  flip_idx?: number[]
}

export function isPoseYamlOptions(
  options: unknown,
): options is PoseYamlOptions {
  const o = options as PoseYamlOptions
  return (
    isString(o.train_path) &&
    isString(o.val_path) &&
    isString(o.test_path) &&
    isNumber(o.n_class) &&
    (o.class_names === undefined || isStringArray(o.class_names)) &&
    isNumber(o.n_keypoints) &&
    typeof o.visibility === 'boolean' &&
    (o.flip_idx === undefined || isNumberArray(o.flip_idx))
  )
}

export function isDetectYamlOptions(
  options: unknown,
): options is DetectYamlOptions {
  const o = options as DetectYamlOptions
  return (
    isString(o.train_path) &&
    isString(o.val_path) &&
    isString(o.test_path) &&
    isNumber(o.n_class) &&
    (o.class_names === undefined || isStringArray(o.class_names))
  )
}

export function parseDataYaml(
  task: 'pose' | 'detect',
  yaml: string,
): PoseYamlOptions | DetectYamlOptions {
  return task === 'pose' ? parsePoseDataYaml(yaml) : parseDetectDataYaml(yaml)
}

function parseDetectDataYaml(yaml: string): DetectYamlOptions {
  const lines = yaml.split('\n')

  const detectOptions: DetectYamlOptions = {
    train_path: findString(lines, 'train'),
    val_path: findString(lines, 'val'),
    test_path: findString(lines, 'test'),
    n_class: findNumber(lines, 'nc'),
    class_names: hasValue(lines, 'names')
      ? findStringArray(lines, 'names')
      : undefined,
  }

  if (!isDetectYamlOptions(detectOptions)) {
    throw new Error('Invalid detect YAML options structure')
  }

  return detectOptions
}

function parsePoseDataYaml(yaml: string): PoseYamlOptions {
  const lines = yaml.split('\n')

  const kpt_shape = findNumberArray(lines, 'kpt_shape')
  const n_keypoints = kpt_shape[0]
  const visibility = kpt_shape[1] === 3

  const poseOptions: PoseYamlOptions = {
    train_path: findString(lines, 'train'),
    val_path: findString(lines, 'val'),
    test_path: findString(lines, 'test'),
    n_class: findNumber(lines, 'nc'),
    class_names: hasValue(lines, 'names')
      ? findStringArray(lines, 'names')
      : undefined,
    keypoint_names: hasValue(lines, '# keypoints')
      ? findStringArray(lines, '# keypoints')
      : undefined,
    n_keypoints,
    visibility,
    flip_idx: hasValue(lines, 'flip_idx')
      ? findNumberArray(lines, 'flip_idx')
      : undefined,
  }

  if (!isPoseYamlOptions(poseOptions)) {
    throw new Error('Invalid pose YAML options structure')
  }

  return poseOptions
}

class YamlBuilder {
  lines: string[] = []

  addLine(line: string) {
    this.lines.push(line)
  }

  toString(): string {
    return this.lines.join('\n').trim() + '\n'
  }
}

function toBaseDataYamlString(options: BaseYamlOptions): YamlBuilder {
  let yaml = new YamlBuilder()
  yaml.addLine(`train: ${options.train_path}`)
  yaml.addLine(`val: ${options.val_path}`)
  yaml.addLine(`test: ${options.test_path}`)
  yaml.addLine(``)
  yaml.addLine(`nc: ${options.n_class} # Number of classes`)

  if (options.class_names) {
    let class_names = options.class_names

    if (class_names.length !== options.n_class) {
      throw new Error(
        `Number of class_names (${class_names.length}) does not match n_class (${options.n_class})`,
      )
    }

    yaml.addLine('# Class names')
    if (class_names.length > 1) {
      yaml.addLine(`names:`)
      for (let i = 0; i < class_names.length; i++) {
        yaml.addLine(`  ${i}: ${class_names[i]}`)
      }
    } else {
      yaml.addLine(`names: ${toString(class_names)}`)
    }
  }

  yaml.addLine(``)

  return yaml
}

export function toDataYamlString(
  task: 'pose' | 'detect',
  options: PoseYamlOptions | DetectYamlOptions,
): string {
  if (task === 'pose') {
    if (!isPoseYamlOptions(options)) {
      throw new Error('Invalid options for pose YAML')
    }
    return toPoseDataYamlString(options)
  } else {
    if (!isDetectYamlOptions(options)) {
      throw new Error('Invalid options for detect YAML')
    }
    return toDetectDataYamlString(options)
  }
}

export function toDetectDataYamlString(options: DetectYamlOptions): string {
  return toBaseDataYamlString(options).toString()
}

export function toPoseDataYamlString(options: PoseYamlOptions): string {
  const { keypoint_names, n_keypoints, visibility, flip_idx } = options

  let yaml = toBaseDataYamlString(options)

  if (keypoint_names) {
    yaml.addLine(`# Keypoints: ${toString(keypoint_names)}`)
  }

  const n_dims = visibility ? 3 : 2
  yaml.addLine(
    `kpt_shape: ${toString([n_keypoints, n_dims])} # [n_keypoints, n_dims]`,
  )

  if (flip_idx && flip_idx.length > 0) {
    yaml.addLine(`flip_idx: ${toString(flip_idx)} # Keypoint flip indexes`)
  }

  return yaml.toString()
}
