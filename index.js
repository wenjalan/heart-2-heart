const tf = require('@tensorflow/tfjs-node');
const fs = require("fs/promises");
const jquery = require('jquery-csv');

// file parameters
const CORPUS = './data/heart.csv';
const MODEL_DIRECTORY = 'file://./model/';
const BUNDLE_DIRECTORY = './model/bundle.json';

// training parameters
const TRAIN_EPOCHS = 50;
const TRAIN_BATCHES = 20;

// main
async function start() {
    // load training data
    const [trainXs, trainYs, testXs, testYs, inputSize] = await getTrainingData(CORPUS);

    // create a model
    const model = createModel(inputSize);
    model.summary();

    // train the model
    await model.fit(trainXs, trainYs, {
        epochs: TRAIN_EPOCHS,
        batchSize: TRAIN_BATCHES,
        shuffle: true,
    });

    // save the model and bundle to disk
    await model.save(MODEL_DIRECTORY);
    // await fs.writeFile(BUNDLE_DIRECTORY, JSON.stringify(metadata));
    console.log('Saved model');

    // evaluate?
    const preds = model.predict(testXs);
    const predsArray = await preds.array();
    const actualArray = await testYs.array();
    for (let i = 0; i < predsArray.length; i++) {
        console.log('predicted: ' + predsArray[i] + ', actual: ' + actualArray[i]);
    }

}

// returns a model to fit data to
function createModel(inputSize) {
    // sequential
    const model = tf.sequential();

    // dense input
    model.add(tf.layers.dense({
        inputShape: [inputSize],
        units: 128,
    }));

    // dense intermediate
    model.add(tf.layers.dense({
        units: 256,
    }));

    // dense output: 0 or 1
    model.add(tf.layers.dense({
        units: 1,
    }));

    // compile
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['accuracy'],
    });

    // return
    return model;
}

// returns training tensors given a corpus
// corpus: the directory of a file to generate training data from
async function getTrainingData(corpus) {
    // log to console
    console.log('Generating training data from corpus', corpus);

    // read in the file, clean it and turn it into an array
    let data = await fs.readFile(corpus, 'utf-8');
    let table = await jquery.toArrays(data);
    // console.log(table);
    console.log('Read ' + table.length + ' rows from file ' + corpus);

    // split training data into headers, train and test
    let headers = table[0];
    let trainData = table.slice(1, table.length - 200);
    let testData = table.slice(table.length - 200, table.length);

    // turn trainData and testData into tensors
    let [trainXs, trainYs] = toTensors(trainData);
    let [testXs, testYs] = toTensors(testData);
    // trainXs.print();
    // trainYs.print();
    // testXs.print();
    // testYs.print();

    // return
    return [trainXs, trainYs, testXs, testYs, headers.length - 1];
}

function toTensors(data) {
    // form inputs and labels
    const inputs = data.map(d => d.slice(0, d.length - 1));
    const labels = data.map(d => d[d.length - 1]);

    // form tensors
    const inputTensor = tf.tensor2d(inputs, [inputs.length, inputs[0].length], 'float32');
    const labelTensor = tf.tensor2d(labels, [labels.length, 1], 'float32');

    // normalize tensors
    const inMax = inputTensor.max(0);
    const inMin = inputTensor.min(0);
    // inMax.print();
    // inMin.print();
    const laMax = labelTensor.max();
    const laMin = labelTensor.min();

    let normalIn = inputTensor.sub(inMin);
    let denom = inMax.sub(inMin);
    normalIn = normalIn.div(denom);
    const normalLa = labelTensor.sub(laMin).div(laMax.sub(laMin));

    // return labels
    return [normalIn, normalLa];
}

// starts the program
start();