export type ImageModelSpec = {
  url: string
  width: number
  height: number
  channels: number
  features: number
  spatial_features?: SpatialFeatures
  spatial_layers?: SpatialLayer[]
}

export type SpatialFeatures = [
  batch: number,
  height: number,
  width: number,
  features: number,
]

export type SpatialLayer = {
  name: string
  layer: number
  shape: SpatialFeatures
}

export const PreTrainedImageModels = {
  mobilenet: {
    // #param, accuracy, and latency see: https://keras.io/api/applications/mobilenet/#mobilenetv3large-function
    'mobilenet-v3-large-100': {
      url: 'https://www.kaggle.com/models/google/mobilenet-v3/TfJs/large-100-224-feature-vector/1' as const,
      width: 224 as const,
      height: 224 as const,
      channels: 3 as const,
      features: 1280 as const,
      spatial_features: [1, 7, 7, 160] as const satisfies SpatialFeatures,
      spatial_layers: [
        {
          name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_2/project/BatchNorm/FusedBatchNormV3',
          layer: 2,
          shape: [1, 56, 56, 24],
        },
        {
          name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_5/project/BatchNorm/FusedBatchNormV3',
          layer: 5,
          shape: [1, 28, 28, 40],
        },
        {
          name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_9/project/BatchNorm/FusedBatchNormV3',
          layer: 9,
          shape: [1, 14, 14, 80],
        },
        {
          name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_11/project/BatchNorm/FusedBatchNormV3',
          layer: 11,
          shape: [1, 14, 14, 112],
        },
        {
          name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_14/project/BatchNorm/FusedBatchNormV3',
          layer: 14,
          shape: [1, 7, 7, 160],
        },
      ] as const satisfies SpatialLayer[],
    },
    'mobilenet-v3-large-75': {
      url: 'https://www.kaggle.com/models/google/mobilenet-v3/TfJs/large-075-224-feature-vector/1' as const,
      width: 224 as const,
      height: 224 as const,
      channels: 3 as const,
      features: 1280 as const,
      spatial_features: [1, 7, 7, 120] as const satisfies SpatialFeatures,
      spatial_layers: [
        {
          name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_2/project/BatchNorm/FusedBatchNormV3',
          layer: 2,
          shape: [1, 56, 56, 24],
        },
        {
          name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_5/project/BatchNorm/FusedBatchNormV3',
          layer: 5,
          shape: [1, 28, 28, 32],
        },
        {
          name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_9/project/BatchNorm/FusedBatchNormV3',
          layer: 9,
          shape: [1, 14, 14, 64],
        },
        {
          name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_11/project/BatchNorm/FusedBatchNormV3',
          layer: 11,
          shape: [1, 14, 14, 88],
        },
        {
          name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_14/project/BatchNorm/FusedBatchNormV3',
          layer: 14,
          shape: [1, 7, 7, 120],
        },
      ] as const satisfies SpatialLayer[],
    },
    'mobilenet-v3-small-100': {
      url: 'https://www.kaggle.com/models/google/mobilenet-v3/TfJs/small-100-224-feature-vector/1' as const,
      width: 224 as const,
      height: 224 as const,
      channels: 3 as const,
      features: 1280 as const,
      spatial_features: [1, 7, 7, 96] as const satisfies SpatialFeatures,
      spatial_layers: [
        {
          name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_2/project/BatchNorm/FusedBatchNormV3',
          layer: 2,
          shape: [1, 28, 28, 24],
        },
        {
          name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_5/project/BatchNorm/FusedBatchNormV3',
          layer: 5,
          shape: [1, 14, 14, 40],
        },
        {
          name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_7/project/BatchNorm/FusedBatchNormV3',
          layer: 7,
          shape: [1, 14, 14, 48],
        },
        {
          name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_10/project/BatchNorm/FusedBatchNormV3',
          layer: 10,
          shape: [1, 7, 7, 96],
        },
      ] as const satisfies SpatialLayer[],
    },
    'mobilenet-v3-small-75': {
      url: 'https://www.kaggle.com/models/google/mobilenet-v3/TfJs/small-075-224-feature-vector/1' as const,
      width: 224 as const,
      height: 224 as const,
      channels: 3 as const,
      features: 1280 as const,
      spatial_features: [1, 7, 7, 72] as const satisfies SpatialFeatures,
      spatial_layers: [
        {
          name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_2/project/BatchNorm/FusedBatchNormV3',
          layer: 2,
          shape: [1, 28, 28, 24],
        },
        {
          name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_5/project/BatchNorm/FusedBatchNormV3',
          layer: 5,
          shape: [1, 14, 14, 32],
        },
        {
          name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_7/project/BatchNorm/FusedBatchNormV3',
          layer: 7,
          shape: [1, 14, 14, 40],
        },
        {
          name: 'StatefulPartitionedCall/StatefulPartitionedCall/predict/MobilenetV3/expanded_conv_10/project/BatchNorm/FusedBatchNormV3',
          layer: 10,
          shape: [1, 7, 7, 72],
        },
      ] as const satisfies SpatialLayer[],
    },
  } satisfies Record<string, ImageModelSpec>,
}
