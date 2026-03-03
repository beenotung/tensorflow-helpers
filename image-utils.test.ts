import { expect } from 'chai'
import { scaleTensor } from './image-utils'
import * as tf from '@tensorflow/tfjs-node'

describe('scaleTensor', () => {
  function test(options: {
    from_range: [min: number, max: number]
    from_pixels: number[]
    to_range: [min: number, max: number]
    to_pixels: number[]
  }) {
    it(`should scale tensor from [${options.from_range}] to [${options.to_range}]`, () => {
      let tensor = tf.tensor(options.from_pixels) as tf.Tensor4D
      let result = scaleTensor({
        tensor,
        fromRange: options.from_range,
        toRange: options.to_range,
      })
      let result_pixels = Array.from(result.dataSync())
      expect(result_pixels).to.have.length(3)
      expect(result_pixels[0]).to.be.closeTo(options.to_pixels[0], 1e-6)
      expect(result_pixels[1]).to.be.closeTo(options.to_pixels[1], 1e-6)
      expect(result_pixels[2]).to.be.closeTo(options.to_pixels[2], 1e-6)
    })
  }
  test({
    from_range: [0, 255],
    from_pixels: [0, 127.5, 255],
    to_range: [0, 1],
    to_pixels: [0, 0.5, 1],
  })
  test({
    from_range: [0, 255],
    from_pixels: [0, 127.5, 255],
    to_range: [-1, 1],
    to_pixels: [-1, 0, 1],
  })
  test({
    from_range: [0, 1],
    from_pixels: [0, 0.5, 1],
    to_range: [0, 255],
    to_pixels: [0, 127.5, 255],
  })
  test({
    from_range: [-1, 1],
    from_pixels: [-1, 0, 1],
    to_range: [0, 255],
    to_pixels: [0, 127.5, 255],
  })
})
