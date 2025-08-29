import { readdirSync, readFileSync } from "fs";
import { join } from "path";
import { calcGridScore, loadModels, NormalizedBox, PixelBox } from "./util";
import * as tf from "@tensorflow/tfjs-node";
import { getImageFeatures } from "../image-features";
import { PreTrainedImageModels } from "../image-model";
import { loadLayersModel, saveModel } from "../model";
import sharp from "sharp";
import fs from "fs";
import { cropAndResizeImageTensor } from "../image-utils";

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
  let c1 = await loadLayersModel({dir:"./saved_models/c1-model"});
  let { baseModel, classifierModel2D: c2 } = await loadModels();

  let trainData = readDataset(trainDatasetDir); //{imagePath, box}
  let testData = readDataset(testDatasetDir);

  compile();
  await train({ epoch: 20, batchSize: 32 });

  /* setup for training */
  function compile() {
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
      let ys = [];

      let { imagePath, box: expected_box } = data;

      let embedding1 = (await getImageFeatures({
        tf,
        imageModel: baseModel,
        image: imagePath,
      })).spatialFeatures;

      let result1 = tf.tidy(()=>{return c1.predict(embedding1) as tf.Tensor});
      let [matrix] = (await result1.array()) as number[][][][]

      let targets = [] as [x: number, y: number][]
      const target_threshold = 0.3

      for (let y = 0; y < matrix.length; y++) {
        for (let x = 0; x < matrix[y].length; x++) {
          let [background, object] = matrix[y][x]
          if (object >= target_threshold) {
            targets.push([x, y])
          }
        }
      }
      if (targets.length === 0) {
        //if no target, use center grid
        targets.push([0,0])
        targets.push([COL-1, ROW-1])
      }
      let target_x = targets.map((t) => t[0])
      let target_y = targets.map((t) => t[1])

      let pixel_expected_box: PixelBox = {
        left: expected_box.x * 224 - (expected_box.width * 224) / 2,
        top: expected_box.y * 224 - (expected_box.height * 224) / 2,
        right: expected_box.x * 224 + (expected_box.width * 224) / 2,
        bottom: expected_box.y * 224 + (expected_box.height * 224) / 2,
      }
      
      let pixel_predicted_box: PixelBox = {
      left: Math.min(...target_x) * 32 ,
      top: Math.min(...target_y) * 32,
      right: (Math.max(...target_x) + 1) * 32 ,
      bottom: (Math.max(...target_y) + 1) * 32,
    }
    let predicted_width = pixel_predicted_box.right - pixel_predicted_box.left
    let predicted_height = pixel_predicted_box.bottom - pixel_predicted_box.top
    
    //rect to square option
      // let left =  Math.min(...target_x) * 32 
      // let top = Math.min(...target_y) * 32
      // let right = (Math.max(...target_x) + 1) * 32 
      // let bottom = (Math.max(...target_y) + 1) * 32
      // let predicted_width = right - left
      // let predicted_height = bottom - top
      // let width_height_diff = predicted_width - predicted_height
      // let pixel_predicted_box: PixelBox

      // if (width_height_diff > 0) {
      //   //width > height, expand height
      //   predicted_height = width_height_diff
      //   top -= width_height_diff / 2
      //   bottom += width_height_diff / 2
      //   if (top < 0) {
      //     bottom -= top
      //     top = 0
      //   }
      //   if (bottom > 224) {
      //     top -= (bottom - 224)
      //     bottom = 224
      //   }
      // }
      // else if (width_height_diff < 0) {
      //   //height > width, expand width
      //   predicted_width -= width_height_diff
      //   left += width_height_diff / 2
      //   right -= width_height_diff / 2
      //   if (left < 0) {
      //     right -= left
      //     left = 0
      //   }
      //   if (right > 224) {
      //     left -= (right - 224)
      //     right = 224
      //   }
      // }
      // pixel_predicted_box = {
      //   left,
      //   top,
      //   right,
      //   bottom,
      // }

      let cropped_img_tensor;
      try {
        // Read and crop the image
        const inputBuffer = fs.readFileSync(imagePath);
        const metadata = await sharp(inputBuffer).metadata();

        const imageWidth = metadata.width ?? 0;
        const imageHeight = metadata.height ?? 0;
        let {data, info} = await sharp(imagePath)
          .extract({
            left: Math.round(pixel_predicted_box.left * imageWidth / 224),   // X coordinate of top-left corner
            top: Math.round(pixel_predicted_box.top * imageHeight / 224),    // Y coordinate of top-left corner
            width: Math.round((pixel_predicted_box.right - pixel_predicted_box.left) * imageWidth / 224),  // Width of cropped area
            height: Math.round((pixel_predicted_box.bottom - pixel_predicted_box.top) * imageHeight / 224), // Height of cropped area
          })
          .raw()
          .toBuffer({ resolveWithObject: true });
        cropped_img_tensor = cropAndResizeImageTensor({
                imageTensor: tf.tensor3d(data, [info.height, info.width, info.channels], "int32"),
                width: 224,
                height: 224,
              })
      } catch (error) {
        console.error("Error cropping image:", error);
      }
      
      let embedding2 = await getImageFeatures({
        tf,
        imageModel: baseModel,
        image: cropped_img_tensor!,
      })
      spatial_features.push(embedding2.spatialFeatures.squeeze([0]));

      for (let y = 0; y < ROW; y++) {
        let xs = [];
        for (let x = 0; x < COL; x++) {
          let grid: PixelBox = {
            left: pixel_predicted_box.left + x * predicted_width / COL,
            top: pixel_predicted_box.top + y * predicted_height / ROW,
            right: pixel_predicted_box.left + (x + 1) * predicted_width / COL,
            bottom: pixel_predicted_box.top + (y + 1) * predicted_height / ROW,
          };
          let score = calcGridScore(grid, pixel_expected_box);
          let object = score;
          let background = 1 - object;
          xs.push([background, object]);
        }
        ys.push(xs);
      }
      ys_batch.push(tf.tensor(ys));
    }
    const xTrain = tf.stack(spatial_features);
    const yTrain = tf.stack(ys_batch);
    
    console.log("finished data process, start training...");

    await c2.fit(xTrain, yTrain, {
      epochs: options.epoch || 20,
      batchSize: options.batchSize || 1,
      //TODO? how to use classWeight? Need or not?
      // classWeight: [
      //   (1 - background_count / total_count) * 2,
      //   (1 - object_count / total_count) * 2,
      // ],
    });
    saveModel({ model: c2, dir: "./saved_models/c2-model" });
    console.log("after train");
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
