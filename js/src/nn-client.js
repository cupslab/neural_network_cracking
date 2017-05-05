function NeuralNetworkClient(callback, configuration) {
  if (!NeuralNetworkWorker.configured) {
    NeuralNetworkWorker.worker.postMessage({
      messageType: "config",
      payload: configuration
    });
    NeuralNetworkWorker.configured = true;
  }

  this.callback = callback;
  // random string id
  this.id = (Math.random()*1e32).toString(36);
  // register this instance
  NeuralNetworkWorker.clients[this.id] = this;
}

// declare shared state for neural network clients
var NeuralNetworkWorker = new Object();
// map of client ids to instances
NeuralNetworkWorker.clients = new Object();
// shared worker
NeuralNetworkWorker.worker = new Worker('worker.min.js');
// configuration flag
NeuralNetworkWorker.configured = false;

NeuralNetworkWorker.onMessageTriggered = function(event) {
  var tag = event.data.tag;
  var client = NeuralNetworkWorker.clients[tag];
  if (typeof client !== "undefined") {
    client.callback(event.data.prediction, event.data.password);
  }
};

NeuralNetworkWorker.worker.onmessage = NeuralNetworkWorker.onMessageTriggered;

NeuralNetworkClient.prototype.query = function(pwd, prefix) {
  NeuralNetworkWorker.worker.postMessage({
    tag : this.id,
    inputData : pwd,
    action : 'total_prob',
    prefix : prefix
  });
};

NeuralNetworkClient.prototype.query_guess_number = function(pwd) {
  NeuralNetworkWorker.worker.postMessage({
    tag : this.id,
    inputData : pwd,
    action : 'guess_number'
  });
};

NeuralNetworkClient.prototype.predict_next = function(pwd) {
  NeuralNetworkWorker.worker.postMessage({
    tag : this.id,
    inputData : pwd,
    action : 'predict_next'
  });
};

NeuralNetworkClient.prototype.raw_predict_next = function(pwd) {
  NeuralNetworkWorker.worker.postMessage({
    tag : this.id,
    inputData : pwd,
    action : 'raw_predict_next'
  });
};

NeuralNetworkClient.prototype.probability_char = function(pwd, next_char) {
  NeuralNetworkWorker.worker.postMessage({
    tag : this.id,
    inputData : pwd,
    action : 'predict_next'
  });
};

global.NeuralNetworkClient = NeuralNetworkClient;
