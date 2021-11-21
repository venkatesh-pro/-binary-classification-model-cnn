import * as tf from '@tensorflow/tfjs'

const modelFunc = (imgData) => {
  const classNames = ['pizza', 'steak']

  const img = new Image()

  // for tfjs model
  img.src = imgData
  img.onload = async () => {
    const tensor = tf.browser.fromPixels(img)
    // changing the tensor into 224*224 size
    let resizedImg = tensor
      .resizeBilinear([224, 224])
      .toFloat()
      .div(tf.scalar(255))
    // to expand the dimention for the batch size
    let expandDimImg = tf.expandDims(resizedImg)
    if (expandDimImg) {
      const model = await tf.loadLayersModel('/model/model.json')
      let predictions = model && model.predict(expandDimImg)
      predictions = predictions.dataSync()

      let out = classNames[Math.round(predictions[0])]
      if (out) return out
    }
  }
  // for tflite model
  img.onload = async () => {
    const tensor = tf.browser.fromPixels(img)
    let img1 = tensor.resizeBilinear([224, 224]).toFloat().div(tf.scalar(255))
    let a = tf.expandDims(img1)
    console.log(a)
    const load_model = async (a) => {
      const tfliteModel = await tflite.loadTFLiteModel(
        '/youtube_tflite_model.tflite'
      )
      const pred = tfliteModel.predict(a)
      const output_values = tf.softmax(pred.arraySync()[0])
      console.log(pred.arraySync()[0])
    }
    load_model(a)
  }
}
export default modelFunc
