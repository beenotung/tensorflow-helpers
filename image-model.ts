export type ImageModelSpec = {
  url: string
  width: number
  height: number
  channels: number
  features: number
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
    },
    'mobilenet-v3-large-75': {
      url: 'https://www.kaggle.com/models/google/mobilenet-v3/TfJs/large-075-224-feature-vector/1' as const,
      width: 224 as const,
      height: 224 as const,
      channels: 3 as const,
      features: 1280 as const,
    },
    'mobilenet-v3-small-100': {
      url: 'https://www.kaggle.com/models/google/mobilenet-v3/TfJs/small-100-224-feature-vector/1' as const,
      width: 224 as const,
      height: 224 as const,
      channels: 3 as const,
      features: 1280 as const,
    },
    'mobilenet-v3-small-75': {
      url: 'https://www.kaggle.com/models/google/mobilenet-v3/TfJs/small-075-224-feature-vector/1' as const,
      width: 224 as const,
      height: 224 as const,
      channels: 3 as const,
      features: 1280 as const,
    },
  },
}
