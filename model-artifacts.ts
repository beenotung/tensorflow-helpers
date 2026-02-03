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
  _model: Model,
): PatchedModelArtifacts {
  let model = _model as any
  if (model.artifacts) {
    return model.artifacts
  }
  if (model.userDefinedMetadata) {
    return model
  }
  if (typeof model.getUserDefinedMetadata === 'function') {
    model.userDefinedMetadata = model.getUserDefinedMetadata()
    return model
  }
  throw new Error('model artifacts not found in ' + model.constructor.name)
}

/**
 * wrapper to unify the LayeredModel and GraphModel
 */
export function exposeModelArtifacts<Model extends object>(model: Model) {
  // Add methods directly to the original model object
  Object.defineProperty(model, 'getArtifacts', {
    value: () => getModelArtifacts(model),
    writable: false,
    enumerable: false,
    configurable: true,
  })

  Object.defineProperty(model, 'classNames', {
    get: () => getModelArtifacts(model).userDefinedMetadata?.classNames,
    set: (value: string[]) => {
      const artifacts = getModelArtifacts(model)
      artifacts.userDefinedMetadata ||= {}
      artifacts.userDefinedMetadata.classNames = value
    },
    enumerable: true,
    configurable: true,
  })

  return model as Model & {
    getArtifacts: () => PatchedModelArtifacts
    classNames?: string[]
  }
}
