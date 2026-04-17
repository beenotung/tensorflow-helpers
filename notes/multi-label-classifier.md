In the classification head, use linear activation, and the last layer do sigmoid instead of softmax.

Be careful if the loss function expect logits or probabilities.

For the sample with unmarked label, don't assume it's all 0, instead use a special value like -1 to represent it.

```ts
function maskedBCEBalanced(pred, target, numClasses, maskValue = -1) {
  const mask = tf.notEqual(target, maskValue)

  let totalLoss = tf.scalar(0) // proper tensor, not JS number
  let classCount = 0

  for (let c = 0; c < numClasses; c++) {
    const classMask = tf.slice(mask, [0, c], [-1, 1])
    const markedCount = tf.sum(tf.cast(classMask, 'float32'))

    if (markedCount > 0) {
      const classPred = tf.booleanMask(pred, classMask)
      const classTarget = tf.booleanMask(target, classMask)
      const classLoss = tf.losses.sigmoidCrossEntropy(classTarget, classPred)
      totalLoss = totalLoss.add(classLoss.div(markedCount))
      classCount++
    }
  }
  return totalLoss.div(classCount) // divide by num classes that had marks
}
```
