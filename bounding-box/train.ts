import { readdirSync, readFileSync } from "fs";
import { join } from "path";
import { calcGridScore, loadModels, NormalizedBox, PixelBox } from "./util";
// import * as tf from "@tensorflow/tfjs";
// import { PreTrainedImageModels, getImageFeatures } from "../browser";
import * as tf from "@tensorflow/tfjs-node";
import { getImageFeatures } from "../image-features";
import { PreTrainedImageModels } from "../image-model";
import { attachClassNames } from "../classifier-utils";
import { saveModel } from "../model";

const trainDatasetDir = "./dataset/train";
const testDatasetDir = "./dataset/test";
let spec = PreTrainedImageModels.mobilenet["mobilenet-v3-large-100"];
// 1x7x7x160
let ROW = spec.spatial_features[1];
let COL = spec.spatial_features[2];

function readDataset(
  datasetDir: string
): { imagePath: string; box: NormalizedBox }[] {
  const imagesDir = join(datasetDir, "images");
  const labelsDir = join(datasetDir, "labels");
  const imageFiles = readdirSync(imagesDir).filter((f) => f.endsWith(".jpg"));
  const data: { imagePath: string; box: NormalizedBox }[] = [];

  for (const imageFile of imageFiles) {
    const baseName = imageFile.replace(/\.jpg$/, "");
    const labelFile = join(labelsDir, baseName + ".txt");
    if (!require("fs").existsSync(labelFile)) continue;
    const labelContent = readFileSync(labelFile, "utf-8").trim();
    const [classId, x, y, width, height] = labelContent
      .split(/\s+/)
      .map(Number);
    if (classId !== 0 || [x, y, width, height].some((n) => isNaN(n))) continue;
    data.push({
      imagePath: join(imagesDir, imageFile),
      box: { x, y, width, height },
    });
  }
  return data;
}

async function main() {
  let { baseModel, classifierModel2D: c1 } = await loadModels();
  let { classifierModel2D: c2 } = await loadModels();

  let trainData = readDataset(trainDatasetDir); //{imagePath, box}
  let testData = readDataset(testDatasetDir);

  compile();
  await train({ epoch: 20, batchSize: 32 });
  // await predict();

  async function predict() {
    //TODO? run model here or in browser/ other file?
  }

  /* setup for training */
  function compile() {
    c1.compile({
      optimizer: "adam",
      loss: tf.metrics.categoricalCrossentropy,
      metrics: [tf.metrics.categoricalAccuracy],
    });
    c2.compile({
      optimizer: "adam",
      loss: tf.metrics.categoricalCrossentropy,
      metrics: [tf.metrics.categoricalAccuracy],
    });
  }
  /* train the model */
  async function train(options: { epoch?: number; batchSize?: number }) {
    console.log("before data process");

    //data process
    let spatial_features = [] as tf.Tensor[]; //as
    let ys_batch = [];

    for (let data of trainData) {
      let background_count = 0;
      let object_count = 0;
      let ys = [];

      let { imagePath, box: expected_box } = data;

      let embedding = await getImageFeatures({
        tf,
        imageModel: baseModel,
        image: imagePath,
      });
      spatial_features.push(embedding.spatialFeatures.squeeze([0]));

      let expected_box_width = expected_box.width * 224;
      let expected_box_height = expected_box.height * 224;
      let expected_box_left = expected_box.x * 224 - expected_box_width / 2;
      let expected_box_top = expected_box.y * 224 - expected_box_height / 2;
      let expected_box_right = expected_box_left + expected_box_width;
      let expected_box_bottom = expected_box_top + expected_box_height;
      let expected_box_pixel: PixelBox = {
        left: expected_box_left,
        top: expected_box_top,
        right: expected_box_right,
        bottom: expected_box_bottom,
      };

      for (let y = 0; y < ROW; y++) {
        let xs = [];
        for (let x = 0; x < COL; x++) {
          let grid: PixelBox = {
            left: x * 32,
            top: y * 32,
            right: (x + 1) * 32,
            bottom: (y + 1) * 32,
          };
          let score = calcGridScore(grid, expected_box_pixel);
          let object = score;
          let background = 1 - object;
          xs.push([background, object]);
          background_count += background;
          object_count += object;
        }
        ys.push(xs);
      }
      ys_batch.push(tf.tensor(ys));
      // console.log({ background_count, object_count });
      // let total_count = background_count + object_count;
    }
    const xTrain = tf.stack(spatial_features);
    const yTrain = tf.stack(ys_batch);
    console.log("finished data process, start training...");

    await c1.fit(xTrain, yTrain, {
      epochs: options.epoch || 20,
      batchSize: options.batchSize || 1,
      //TODO? how to use classWeight? Need or not?
      // classWeight: [
      //   (1 - background_count / total_count) * 2,
      //   (1 - object_count / total_count) * 2,
      // ],
    });
    saveModel({ model: c1, dir: "./saved_models/c1-model" });
    console.log("after train");
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
