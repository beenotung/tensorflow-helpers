/**
 * A factor to give larger hidden layer size for complex tasks:
 * - 1 for easy tasks
 * - 2-3 for medium difficulty tasks
 * - 4-5 for complex tasks
 *
 * Remark: giving too high difficulty may result in over-fitting.
 * */
export type Difficulty = number

let defaultDifficulty = 1

/**
 * Formula `hiddenSize = difficulty * sqrt(inputSize * outputSize)`
 *
 * @example
 * let embeddingFeatures = 1280
 * let classes = 10
 * let difficulty = 2
 * let hiddenSize = calcHiddenLayerSize({
 *   inputSize: embeddingFeatures,
 *   outputSize: classes,
 *   difficulty,
 * })
 * // hiddenSize = 227
 */
export function calcHiddenLayerSize(options: {
  inputSize: number
  outputSize: number
  difficulty?: Difficulty
}) {
  let { inputSize, outputSize, difficulty = defaultDifficulty } = options
  let hiddenSize = difficulty * Math.sqrt(inputSize * outputSize)
  return Math.ceil(hiddenSize)
}

/**
 * Inject one or more hidden layers that's having large gap between input size and output size.
 *
 * @example
 * let embeddingFeatures = 1280
 * let classes = 10
 * let difficulty = 1.5
 * let numberOfHiddenLayers = 2
 * let layers = [embeddingFeatures, classes]
 * injectHiddenLayers({ layers, difficulty, numberOfHiddenLayers })
 * // layers = [1280, 700, 170, 10]
 */
export function injectHiddenLayers(options: {
  layers: number[]
  difficulty?: Difficulty
  numberOfHiddenLayers?: number
}) {
  let { layers, difficulty, numberOfHiddenLayers } = options
  if (layers.length < 2) {
    throw new Error('layers must have at least 2 layers, for input and output')
  }

  function injectLayer() {
    let { index, gap } = layers
      .map((_, index) => {
        if (index == 0) return null
        let inputSize = layers[index - 1]
        let outputSize = layers[index]
        let gap = Math.abs(outputSize - inputSize)
        return { index, gap }
      })
      .filter(entry => entry != null)
      .sort((a, b) => b.gap - a.gap)[0]
    let inputSize = layers[index - 1]
    let outputSize = layers[index]
    let hiddenSize = calcHiddenLayerSize({
      inputSize,
      outputSize,
      difficulty,
    })
    layers.splice(index, 0, hiddenSize)
  }

  if (!numberOfHiddenLayers) {
    injectLayer()
    return
  }

  if (numberOfHiddenLayers < 0) {
    throw new Error('numberOfHiddenLayers must be positive')
  }

  while (layers.length - 2 < numberOfHiddenLayers) {
    injectLayer()
  }
}

export type ImageEmbeddingOptions = {
  expandAnimations?: boolean
  /** default: false */
  squeeze?: boolean
}
