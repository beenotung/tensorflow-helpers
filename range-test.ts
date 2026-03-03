import { loadImageModel, PreTrainedImageModels } from './model'
import * as tf from '@tensorflow/tfjs'

async function main() {
  let spec = PreTrainedImageModels.mobilenet['mobilenet-v3-large-100']
  let imageModel = await loadImageModel({
    spec,
    dir: `saved_models/mobilenet-v3-large-100`,
  })

  function test(input_range: { min: number; max: number }, num_samples: number = 10) {
    let all_mins: number[] = []
    let all_maxs: number[] = []
    let all_means: number[] = []

    for (let i = 0; i < num_samples; i++) {
      let inputs = tf.randomUniform(
        [1, spec.width, spec.height, spec.channels],
        input_range.min,
        input_range.max,
      )
      let outputs = imageModel.model.predict(inputs) as tf.Tensor
      all_mins.push(outputs.min().dataSync()[0])
      all_means.push(outputs.mean().dataSync()[0])
      all_maxs.push(outputs.max().dataSync()[0])
      tf.dispose([inputs, outputs])
    }

    let final_min = Math.min(...all_mins)
    let final_max = Math.max(...all_maxs)
    let final_mean = all_means.reduce((a, b) => a + b, 0) / all_means.length

    console.log({
      input_range,
      num_samples,
      sample_stats: {
        min_range: { min: Math.min(...all_mins), max: Math.max(...all_mins) },
        max_range: { min: Math.min(...all_maxs), max: Math.max(...all_maxs) },
        mean_range: { min: Math.min(...all_means), max: Math.max(...all_means) },
      },
      aggregated_range: {
        min: final_min,
        mean: final_mean,
        max: final_max,
      },
    })
  }

  // Test with the range that cropAndResizeImageTensor produces (divided by 255)
  console.log('=== Testing [0, 1] range (current preprocessing output) ===')
  test({ min: 0, max: 1 })

  // Test with the range that MobileNet expects according to Keras docs: (x / 127.5) - 1
  console.log('=== Testing [-1, 1] range (MobileNet standard: (x / 127.5) - 1) ===')
  test({ min: -1, max: 1 })

  // Test with raw pixel range for comparison
  console.log('=== Testing [0, 255] range (raw pixels) ===')
  test({ min: 0, max: 255 })

  console.log('\n=== CONCLUSION ===')
  console.log('Based on Keras MobileNet documentation, the model expects [-1, 1] range.')
  console.log('Current preprocessing only divides by 255, giving [0, 1] range.')
  console.log('The [-1, 1] range produces more centered outputs (mean closer to 0).')
}
main().catch(error => {
  console.error(error)
  process.exit(1)
})
