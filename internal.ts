import { ModelArtifacts, ModelJSON } from '@tensorflow/tfjs-core/dist/io/types'
import { exposeModelArtifacts, PatchedModelArtifacts } from './model-artifacts'

export type SavedModelJSON = PatchedModelArtifacts & {
  /** @deprecated use userDefinedMetadata.classNames instead */
  classNames?: string[]
}

export function patchLoadedModelJSON(model: SavedModelJSON): boolean {
  let changed = false

  // recover weightsManifest from inline weightData and weightSpecs
  if (model.weightData && model.weightSpecs && !model.weightsManifest) {
    let paths: string[] = []
    let weightData = model.weightData
    if (!Array.isArray(weightData)) {
      weightData = [weightData]
    }
    let n = weightData.length
    for (let i = 0; i < n; i++) {
      let filename = `weight-${i}.bin`
      paths.push(filename)
    }
    model.weightsManifest = [{ paths, weights: model.weightSpecs }]
    delete model.weightData
    delete model.weightSpecs
    changed = true
  }

  // move the classNames to userDefinedMetadata
  if (model.classNames) {
    model.userDefinedMetadata ||= {}
    model.userDefinedMetadata.classNames = model.classNames
    delete model.classNames
    changed = true
  }

  return changed
}

/**
 * @deprecated use ModelJSON.userDefinedMetadata.classNames instead
 */
export type ModelArtifactsWithClassNames = ModelArtifacts & {
  classNames?: string[]
}

export function checkClassNames(
  modelArtifact: ModelJSON,
  classNames: undefined | string[],
): undefined | string[] {
  let classNamesInMetadata = modelArtifact.userDefinedMetadata?.classNames

  if (classNamesInMetadata) {
    if (!Array.isArray(classNamesInMetadata)) {
      throw new Error('classNames in userDefinedMetadata is not an array')
    }
    if (typeof classNamesInMetadata[0] !== 'string') {
      throw new Error(
        'classNames in userDefinedMetadata is not an array of strings',
      )
    }
  }

  if (classNames && classNamesInMetadata) {
    let expected = JSON.stringify(classNames)
    let actual = JSON.stringify(classNamesInMetadata)
    if (actual !== expected) {
      throw new Error(
        `classNames mismatch, expected: ${expected}, actual: ${actual}`,
      )
    }
  }

  return !classNames && classNamesInMetadata
    ? (classNamesInMetadata as string[])
    : classNames
}

type AnyModel = { save: Function }

export function attachClassNames<Model extends AnyModel>(
  _model: Model,
  classNames: undefined | string[],
) {
  let model = exposeModelArtifacts(_model)

  if (classNames) {
    let artifacts = model.getArtifacts()
    artifacts.userDefinedMetadata ||= {}
    artifacts.userDefinedMetadata.classNames = classNames
  }

  return model
}
