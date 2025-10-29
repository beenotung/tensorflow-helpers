#!/usr/bin/env node

import * as tf from '@tensorflow/tfjs'
import { mkdirSync } from 'fs'
import { dirname, join } from 'path'
import { loadGraphModel, loadLayersModel, saveModel } from './model'
import { getModelArtifacts } from './model-artifacts'

let helpMessage = `
Usage: download-tfjs-model <source> <output-dir>

Download and save TensorFlow.js models for use in browser or Node.js.
Supports URLs (TensorFlow Hub, Kaggle) and local model files/directories.

Examples:

  # Download MobileNet V3
  download-tfjs-model https://www.kaggle.com/models/google/mobilenet-v3/TfJs/large-100-224-feature-vector/1 ./browser-models/mobilenet-v3-large-100

  # Download MobileNet V2
  download-tfjs-model https://www.kaggle.com/models/google/mobilenet-v2/TfJs/035-128-feature-vector/3 ./browser-models/mobilenet-v2-035

  # Convert local model
  download-tfjs-model ./hub-models/mobilenet-v2-035-128-feature-vector ./browser-models/mobilenet-v2-035
`.trim()

async function main() {
  const args = process.argv.slice(2)

  if (args.includes('--help') || args.includes('-h')) {
    console.log(helpMessage)
    process.exit(0)
  }

  if (args.length != 2) {
    console.error(helpMessage)
    process.exit(1)
  }

  const [source, outputDir] = args

  try {
    console.log(`Loading model from: ${source}`)
    const model = await loadModel(source)
    let artifacts = getModelArtifacts(model)
    artifacts.userDefinedMetadata ||= {}
    artifacts.userDefinedMetadata.source = source

    console.log(`Saving model to: ${outputDir}`)
    mkdirSync(outputDir, { recursive: true })
    await saveModel({ model, dir: outputDir })

    console.log('✅ Model downloaded and saved successfully!')
  } catch (error) {
    console.error('❌ Error:', error)
    process.exit(1)
  }
}

async function loadModel(
  source: string,
): Promise<tf.GraphModel | tf.LayersModel> {
  if (!source.startsWith('http') && !source.endsWith('.json')) {
    source = join(source, 'model.json')
  }
  if (!source.includes('://')) {
    source = `file://${source}`
  }

  try {
    // try layered model
    if (source.startsWith('file://')) {
      return await loadLayersModel({ dir: toDir(source) })
    } else {
      return await tf.loadLayersModel(source, { fromTFHub: true })
    }
  } catch {
    // try graph model
    if (source.startsWith('file://')) {
      return await loadGraphModel({ dir: toDir(source) })
    } else {
      return await tf.loadGraphModel(source, { fromTFHub: true })
    }
  }
}

function toDir(source: string): string {
  source = source.replace('file://', '')
  return dirname(source)
}

main()
