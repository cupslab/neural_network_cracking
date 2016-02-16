function NeuralNetworkClient(callback) {
  this.callback = callback;
  this.worker = new Worker('worker.min.js');
  this.worker.onmessage = this.onMessageTriggered.bind(this);
}

NeuralNetworkClient.prototype.onMessageTriggered = function(event) {
  this.callback(event.data.prediction, event.data.password);
};

NeuralNetworkClient.prototype.query = function(pwd, prefix) {
  this.worker.postMessage({
    inputData : pwd,
    action : 'total_prob',
    prefix : prefix
  });
};

NeuralNetworkClient.prototype.query_guess_number = function(pwd) {
  this.worker.postMessage({
    inputData : pwd,
    action : 'guess_number'
  });
};

NeuralNetworkClient.prototype.predict_next = function(pwd) {
  this.worker.postMessage({
    inputData : pwd,
    action : 'predict_next'
  });
};

NeuralNetworkClient.prototype.raw_predict_next = function(pwd) {
  this.worker.postMessage({
    inputData : pwd,
    action : 'raw_predict_next'
  });
};

NeuralNetworkClient.prototype.probability_char = function(pwd, next_char) {
  this.worker.postMessage({
    inputData : pwd,
    action : 'predict_next'
  });
};

global.NeuralNetworkClient = NeuralNetworkClient;
