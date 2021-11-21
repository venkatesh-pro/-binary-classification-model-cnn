import React, { useState } from 'react'
import modelFunc from '../helper/modelFunc'
import * as tf from '@tensorflow/tfjs'
import styles from './index.module.css'
const Index = () => {
  const [imga, setImga] = useState('')
  const [output, setOutput] = useState('')
  const fileChangeEvent = (e) => {
    const file = e.target.files[0]

    const reader = new window.FileReader()

    reader.readAsDataURL(file)

    reader.onload = function () {
      setImga(reader.result)

      const classNames = ['pizza', 'steak']

      const img = new Image()

      // for tfjs model
      img.src = reader.result
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
          const model = await tf.loadLayersModel(
            'https://raw.githubusercontent.com/venkatesh-pro/All-Model/main/binary%20classification%20model/model.json'
          )
          let predictions = model && model.predict(expandDimImg)
          predictions = predictions.dataSync()

          let out = classNames[Math.round(predictions[0])]
          console.log(out)
          setOutput(out)
        }
      }
    }
  }
  return (
    <div className={styles.container}>
      <label>
        Upload Image
        <input type='file' hidden onChange={fileChangeEvent} accept='image/*' />
      </label>
      <img src={imga} />
      {output && <span>The predicted output is {output}</span>}
    </div>
  )
}

export default Index
