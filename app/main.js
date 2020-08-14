const tf = require("@tensorflow/tfjs");

let model

const loadTfModel = async () => {
    console.log('Loading model...')
    model = await tf.loadLayersModel('./models/model.json');
    console.log('Successfully loaded model');
}

loadTfModel().then(r => {console.log("Done, result = " + r)});
