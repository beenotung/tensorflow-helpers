import { readFile, writeFile } from 'fs/promises'

type ModelJSON = {
  modelTopology: {
    node: Array<{
      name: string
      op?: string
      input?: string[]
    }>
  }
}

/**
 * to be pasted to https://mermaid.live/edit
 */
export function toMermaidChartText(model: ModelJSON) {
  let text = `
stateDiagram-v2
    A --> B1
    A --> B2
    B1 --> C
    B2 --> C
`.trim()
  return text
}

async function test() {
  let file = 'saved_model/mobilenet-v3-large-100/model.json'
  let content = await readFile(file, 'utf-8')
  let model = JSON.parse(content) as ModelJSON
  let text = toMermaidChartText(model)
  await writeFile('chart.txt', text)
  // to be
}
test().catch(e => console.error(e))
