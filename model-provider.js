var jQueryScript = document.createElement('script');  
jQueryScript.setAttribute('src','https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.1.2/dist/tf.min.js');
document.head.appendChild(jQueryScript);

function getTensorFlowJsModel(){
    const model = tf.sequential();    
    model.add(tf.layers.dense({units: 50, inputShape: [1], activation: 'relu', kernel_regularizer: 'l2'}));
    model.add(tf.layers.dense({units: 500, activation: 'relu', kernel_regularizer: 'l2'}));
    model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
    model.compile({loss: 'binaryCrossentropy', optimizer: 'rmsprop', metrics: 'accuracy'});     
    return model;
}

function getEpochs(){
    return 500;
}