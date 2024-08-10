# tensorflow-helpers

Helper functions to use tensorflow in nodejs / browser

[![npm Package Version](https://img.shields.io/npm/v/tensorflow-helpers)](https://www.npmjs.com/package/tensorflow-helpers)

## Features

- Support transfer learning and continuous learning
- Custom image classifier using embedding features from pre-trained image model
- Correctly save/load model on filesystem[1]
- Load image file into tensor with resize and crop
- List varies pre-trained models (url, image dimension, embedding size)
- Typescript support
- Isomorphic package: works in Node.js and browsers
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

## License

This project is licensed with [BSD-2-Clause](./LICENSE)

This is free, libre, and open-source software. It comes down to four essential freedoms [[ref]](https://seirdy.one/2021/01/27/whatsapp-and-the-domestication-of-users.html#fnref:2):

- The freedom to run the program as you wish, for any purpose
- The freedom to study how the program works, and change it so it does your computing as you wish
- The freedom to redistribute copies so you can help others
- The freedom to distribute copies of your modified versions to others
