const tf = require('@tensorflow/tfjs-node');
const prompt = require('prompt-sync')();

const MODEL_DIRECTORY = 'file://./model/model.json';

async function test() {
    // load model from disk
    const model = await tf.loadLayersModel(MODEL_DIRECTORY);
    model.summary();

    // make predictions
    const patientInfo = [];
    console.log('ENTER PATIENT INFORMATION ===');
    patientInfo[0] = prompt('> AGE: ');
    patientInfo[1] = prompt('> RESTING BP: ');
    patientInfo[2] = prompt('> CHOLESTEROL: ');
    patientInfo[3] = prompt('> FASTING BS: ');
    patientInfo[4] = prompt('> MAX HR: ');
    patientInfo[5] = prompt('> OLDPEAK: ');

    // make a tensor
    const input = tf.tensor2d(patientInfo, [1, patientInfo.length], 'float32');
    const pred = model.predict(input);
    const predPercent = pred.arraySync()[0][0];
    console.log('HEART DISEASE PREDICTION: ' + predPercent);
}

test();