import * as tf from '@tensorflow/tfjs'

export type ClassifierModelSpec = {
  embeddingFeatures: number
  hiddenLayers?: number[]
  classes: number
  classNames?: string[]
}

export function createImageClassifier(spec: ClassifierModelSpec) {
  let { hiddenLayers, classNames } = spec

  if (spec.classes < 2) {
    throw new Error('image classifier must be at least 2 classes')
  }
  if (classNames && classNames.length !== spec.classes) {
    throw new Error('classNames length mismatch')
  }

  let classifierModel = tf.sequential()
  classifierModel.add(
    tf.layers.inputLayer({ inputShape: [spec.embeddingFeatures] }),
  )
  classifierModel.add(tf.layers.dropout({ rate: 0.5 }))
  if (hiddenLayers) {
    for (let i = 0; i < hiddenLayers.length; i++) {
      classifierModel.add(
        tf.layers.dense({ units: hiddenLayers[i], activation: 'gelu' }),
      )
      classifierModel.add(tf.layers.dropout({ rate: 0.5 }))
    }
  }
  classifierModel.add(
    tf.layers.dense({ units: spec.classes, activation: 'softmax' }),
  )

  return attachClassNames(classifierModel, classNames)
}

export type ClassificationResult = {
  label: string
  /** @description between 0 to 1 */
  confidence: number
}

export function getClassCount(shape: tf.Shape | tf.Shape[]): number {
  for (;;) {
    let value = shape[0]
    for (let i = 1; i < shape.length; i++) {
      value = shape[i] || value
    }
    if (Array.isArray(value)) {
      shape = value
      continue
    }
    if (!value) {
      throw new Error('failed to get class count')
    }
    return value
  }
}

export function topClassifyResult(
  items: ClassificationResult[],
): ClassificationResult {
  let idx = 0
  let max = items[idx]
  for (let i = 1; i < items.length; i++) {
    let item = items[i]
    if (item.confidence > max.confidence) {
      max = item
    }
  }
  return max
}

/**
 * @description the values is returned as is.
 * It should has be applied softmax already.
 * */
export function mapWithClassName(
  classNames: string[],
  values: ArrayLike<number>,
  options?: {
    sort?: boolean
  },
): ClassificationResult[] {
  let result = new Array(classNames.length)
  for (let i = 0; i < classNames.length; i++) {
    result[i] = {
      label: classNames[i],
      confidence: values[i],
    }
  }
  if (options?.sort) {
    result.sort((a, b) => b.confidence - a.confidence)
  }
  return result
}

export function calcClassWeight(options: {
  classes: number
  classCounts: number[]
}) {
  let total = options.classCounts.reduce((acc, c) => acc + c, 0)
  let classWeights = options.classCounts.map(
    count => total / options.classes / count,
  )
  return classWeights
}

export type IOHandler = Exclude<Parameters<tf.GraphModel['save']>[0], string>
export type ModelArtifacts = Parameters<Required<IOHandler>['save']>[0]
export type SaveResult = ReturnType<Required<IOHandler>['save']>

export type ModelArtifactsWithClassNames = ModelArtifacts & {
  classNames?: string[]
}

export function checkClassNames(
  modelArtifact: { classNames?: string[] },
  classNames: undefined | string[],
): undefined | string[] {
  if (classNames && modelArtifact.classNames) {
    let expected = JSON.stringify(classNames)
    let actual = JSON.stringify(modelArtifact.classNames)
    if (actual !== expected) {
      throw new Error(
        `classNames mismatch, expected: ${expected}, actual: ${actual}`,
      )
    }
  }
  return !classNames && modelArtifact.classNames
    ? modelArtifact.classNames
    : classNames
}

export function attachClassNames<Model extends object>(
  model: Model,
  classNames: undefined | string[],
): Model & { classNames?: string[] } {
  return classNames ? Object.assign(model, { classNames }) : model
}
