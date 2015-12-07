'use strict';

importScripts('neocortex.min.js');

var nn = new NeuralNet({
  modelFilePath: './js_model_small.json',
  arrayType: 'float32',
  useGPU: true
});

var onLoadMsgs = [];
onmessage = function(e) {
    onLoadMsgs.push(e);
};

nn.init().then(function() {
    console.log('Worker loaded');
    function handleMsg(e) {
        self.postMessage({
            prediction : nn.predict(e.data.inputData)
        });
    }
    onLoadMsgs.forEach(handleMsg);
    self.onmessage = handleMsg;
}).catch(function(error) {
    console.error(error);
});
