# tensorflow-helpers

Helper functions to use tensorflow in nodejs

[![npm Package Version](https://img.shields.io/npm/v/tensorflow-helpers)](https://www.npmjs.com/package/tensorflow-helpers)

## Features

- Support transfer learning and continuous learning
- Custom image classifier using embedding features from pre-trained image model
- Correctly save/load model on filesystem[1]
- Load image file into tensor with resize and crop
- List varies pre-trained models (url, image dimension, embedding size)
- Typescript support
- Works with plain Javascript, Typescript is not mandatory

[1]: The built-in `tf.loadGraphModel()` cannot load the model saved by `model.save()`

## Installation

```bash
npm install tensorflow-helpers
```

You can also install `tensorflow-helpers` with [pnpm](https://pnpm.io/), [yarn](https://yarnpkg.com/), or [slnpm](https://github.com/beenotung/slnpm)

## Usage Example

```typescript
import {} from 'tensorflow-helpers'
```

## Typescript Signature

```typescript

```
