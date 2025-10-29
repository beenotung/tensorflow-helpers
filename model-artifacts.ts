import { ModelArtifacts, ModelJSON } from '@tensorflow/tfjs-core/dist/io/types'

export type PatchedModelArtifacts = ModelJSON &
  Pick<ModelArtifacts, 'weightData' | 'weightSpecs'> & {
    userDefinedMetadata?: {
      classNames?: string[]
    }
  }

export type ModelWithArtifacts<Model extends object> = Model & {
  artifacts: PatchedModelArtifacts
}

export function getModelArtifacts<Model extends object>(
  model: Model,
): PatchedModelArtifacts {
  let artifacts = (model as any).artifacts
  if (!artifacts) {
    throw new Error('model artifacts not found in ' + model.constructor.name)
  }
  return artifacts
}

export function exposeModelArtifacts<Model extends object>(model: Model) {
  let artifacts = getModelArtifacts(model)
  return Object.assign(model, {
    getArtifacts: () => artifacts,
    get classNames() {
      return artifacts.userDefinedMetadata?.classNames
    },
  })
}
