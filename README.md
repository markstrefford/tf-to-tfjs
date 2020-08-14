# Converting TF model to TF.js

This repo walks through the approach of migrating a custom model from Tensorflow to tensorflow.js

Approach:

1) Use transfer learning from mobilenet to a custom model
1) Convert to tf.js layers model
1) Deploy to the browser

### Current issues

This current approach is giving the following error when loading the converted model using tf.js:

```
Uncaught (in promise) Error: Unknown layer: Functional. This may be due to one of the following reasons:
1. The layer is defined in Python, in which case it needs to be ported to TensorFlow.js or your JavaScript code.
2. The custom layer is defined in JavaScript, but is not registered properly with tf.serialization.registerClass().
    at new ValueError (index.js:58562)
    at deserializeKerasObject (index.js:58821)
    at deserialize (index.js:63305)
    at index.js:68775
    at step (index.js:58480)
    at Object.next (index.js:58461)
    at fulfilled (index.js:58451)
```
