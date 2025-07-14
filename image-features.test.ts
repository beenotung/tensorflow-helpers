import * as tf from '@tensorflow/tfjs-node'
import { Tensor } from '@tensorflow/tfjs'
import { getLastSpatialNodeName } from './spatial-utils'
import { PreTrainedImageModels } from './image-model'
import { loadImageModel } from './model'
import { writeFileSync } from 'fs'
import { getImageFeatures } from './image-features'

async function main() {
  for (let [name, spec] of Object.entries(PreTrainedImageModels.mobilenet)) {
    let imageModel = await loadImageModel({
      spec,
      dir: `saved_model/${name}`,
    })
    let image = 'image.jpg'
    let features = await getImageFeatures({ tf, imageModel, image })

    writeFileSync(
      `${name}-data.json`,
      JSON.stringify(
        {
          spatial: features.spatialFeatures.arraySync(),
          pooled: features.pooledFeatures.arraySync(),
        },
        null,
        2,
      ),
    )
  }
}
main().catch(e => console.error(e))
