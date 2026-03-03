import * as tf from '@tensorflow/tfjs'
import { attachClassNames } from './internal'

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
    tf.layers.dense({ units: spec.classes, activation: 'linear' }),
  )

  return attachClassNames(classifierModel, classNames)
}

export type ClassificationOptions = {
  /** default: true */
  applySoftmax?: boolean
  /** default: false */
  squeeze?: boolean
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

export let calcClassWeight = calcInverseFrequencyClassWeight

export function calcInverseFrequencyClassWeight(options: {
  classes: number
  classCounts: number[]
}) {
  const { classes, classCounts } = options

  if (classCounts.length !== classes) {
    throw new Error(
      `classCounts length (${classCounts.length}) must match classes (${classes})`,
    )
  }

  const total = classCounts.reduce((acc, c) => acc + c, 0)

  if (total === 0) {
    throw new Error('classCounts cannot be all zero')
  }

  const classWeights = classCounts.map(count => {
    if (count <= 0) {
      throw new Error(`class count must be positive, got ${count}`)
    }
    return total / classes / count
  })

  return classWeights
}

export function calcEffectiveClassWeight(options: {
  classCounts: number[]
  beta?: number // e.g. 0.9999
}) {
  const { beta = 0.9999 } = options
  const n = options.classCounts.length
  const weights = new Array(n)

  // calculate the effective class weight (inverse frequency based on effective sample size)
  for (let i = 0; i < n; i++) {
    const count = options.classCounts[i]
    const effective = (1 - Math.pow(beta, count)) / (1 - beta)
    weights[i] = 1 / effective
  }

  // find the median weight
  const sortedWeights = weights.sort((a, b) => a - b)
  let mid = Math.floor(n / 2)
  if (n % 2 === 0) {
    mid = (sortedWeights[mid] + sortedWeights[mid + 1]) / 2
  } else {
    mid = sortedWeights[mid]
  }

  // normalize the weights by the median weight
  for (let i = 0; i < n; i++) {
    weights[i] = weights[i] / mid
  }

  return weights
}

async function main() {
  let classCounts = [50, 70, 30, 1000]
  let weights = calcEffectiveClassWeight({
    classCounts,
  })
  console.log('classCounts:', classCounts)
  console.log('weights:', weights)
  let total_samples = classCounts.reduce((acc, c) => acc + c, 0)
  console.log(
    'average weight:',
    weights.reduce((acc, w, i) => acc + w * classCounts[i], 0) / total_samples,
  )
}
if (require.main === module) {
  main().catch(error => {
    console.error(error)
    process.exit(1)
  })
}
