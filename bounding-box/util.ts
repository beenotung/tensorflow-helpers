import { PreTrainedImageModels } from "../image-model";
import { loadImageModel } from "../model";
import * as tf from "@tensorflow/tfjs-node";

let spec = PreTrainedImageModels.mobilenet["mobilenet-v3-large-100"];

export type NormalizedBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type PixelBox = {
  left: number;
  top: number;
  right: number;
  bottom: number;
};

export async function loadModels() {
  /* image model: 224x224x3 -> 7x7x160 */

  let baseModel = await loadImageModel({
    dir: "./saved_models/base_model",
    spec,
  });

  let hiddenLayerSize = 32;
  let outputSize = 2;

  let classifierModel = tf.sequential();
  /* classifier model: 160 features -> 2 (background or object) */
  classifierModel.add(
    tf.layers.inputLayer({
      inputShape: [spec.spatial_features.slice().pop()!],
    })
  );
  classifierModel.add(tf.layers.dropout({ rate: 0.5 }));
  /* hidden layer */
  classifierModel.add(
    tf.layers.dense({ units: hiddenLayerSize, activation: "gelu" })
  );
  classifierModel.add(tf.layers.dropout({ rate: 0.5 }));
  /* output layer */
  classifierModel.add(tf.layers.dense({ units: 2, activation: "softmax" }));

  /*2d*/
  let input = tf.input({ shape: spec.spatial_features.slice(1) });
  let hidden = tf.layers
    .conv2d({
      filters: hiddenLayerSize,
      kernelSize: 1,
      activation: "gelu",
    })
    .apply(input, ["spatial_features_to_hidden"]) as tf.SymbolicTensor;
  let output = tf.layers
    .conv2d({
      filters: outputSize,
      kernelSize: 1,
      activation: "softmax",
    })
    .apply(hidden, ["hidden_to_output"]) as tf.SymbolicTensor;
  let classifierModel2D = tf.model({ inputs: input, outputs: output });

  return { baseModel, classifierModel, classifierModel2D };
}

let grid_score_threshold = 1 / 3;

/* output 0 or 1 exactly */
export function calcGridScore(grid: PixelBox, expected_box: PixelBox) {
  // if the expected box is inside the grid, return 1
  if (isInside(expected_box, grid) || isInside(grid, expected_box)) {
    return 1;
  }

  // calculate the IOU, return 1 if > threshold
  let iou = calcIOU(grid, expected_box);
  // console.log({ iou, grid_score_threshold })
  return iou >= grid_score_threshold ? 1 : 0;
}

function isInside(inner: PixelBox, outer: PixelBox) {
  return (
    inner.left >= outer.left &&
    inner.right <= outer.right &&
    inner.top >= outer.top &&
    inner.bottom <= outer.bottom
  );
}

function calcIOU(grid_box: PixelBox, expected_box: PixelBox): number {
  /* overlap region */
  let left = Math.max(grid_box.left, expected_box.left);
  let right = Math.min(grid_box.right, expected_box.right);
  let top = Math.max(grid_box.top, expected_box.top);
  let bottom = Math.min(grid_box.bottom, expected_box.bottom);
  let width = right - left;
  let height = bottom - top;
  if (width <= 0 || height <= 0) return 0;
  let area_overlap = width * height;

  /* union region */
  let area_grid =
    (grid_box.right - grid_box.left) * (grid_box.bottom - grid_box.top);
  let area_expected =
    (expected_box.right - expected_box.left) *
    (expected_box.bottom - expected_box.top);

  return Math.max(area_overlap / area_grid, area_overlap / area_expected);
}
