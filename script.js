// consts
const EPOCHS = 10;
const BATCH_SIZE = 10;
// todo: figure out how to save a model to the browser
const MODEL_SAVE_URI = 'todo: figure this one out';

// start
async function start() {
    // log
    console.log('starting!');
    
    // get training data from Data
    const [trainXs, trainYs, testXs, testYs] = await loadData();

    // make a model
    const model = createModel();
    model.summary();

    // train the model
    await model.fit(trainXs, trainYs, {
        epochs: EPOCHS,
        batchSize: BATCH_SIZE,
    });

    // save the model
    await model.save(MODEL_SAVE_URI);

    // todo: evaluate the model

}

// returns tensors for training and testing
// returned object should contain the following:
// - trainXs: a Tensor containing training inputs
// - trainYs: a Tensor containing training labels
// - testXs: a Tensor contianing testing inputs
// - testYs: a Tensor contianing testing labels
async function loadData() {
    // log
    console.log('loading data!');

    // todo: write stuff here to load data from heart clinic spreadsheet

    // return training and test tensors
    return [trainXs, trainYs, testXs, testYs];
}

// returns a model to train the training data on
function createModel() {
    const model = tf.sequential();

    // dense layer
    model.add(tf.layers.dense({
        // todo: figure out the input shape of the dat 
        inputShape: [1],
        units: 128
    }));

    // middle dense layer because why not
    model.add(tf.layers.dense({
        units: 256,
        activation: 'softmax',
    }));

    // add an output layer
    model.add(tf.layers.dense({
        units: [1],
        activation: 'softmax',
    }));

    // compile and return
    model.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError',
        metrics: ['accuracy'],
    });

    // return model
    return model;
}

// run
start();