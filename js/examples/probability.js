var INPUT_CLASS = 'nn-password';
var DISPLAY_CLASS = 'nn-display';
var pwd_input;

function PwdInput(elem, callback) {
  this.elem = elem;
  this.client = new NeuralNetworkClient(callback);
  this.elem.oninput = this.onChangeTriggered.bind(this);
}

PwdInput.prototype.onChangeTriggered = function(event) {
  this.client.query_guess_number(this.elem.value);
};

var $ = function(query) {
  return document.getElementsByClassName(query)[0];
};

function init() {
  console.log('Loading scripts');
  var display_next = $(DISPLAY_CLASS);
  pwd_input = new PwdInput($(INPUT_CLASS), function(guess_number, pwd) {
    display_next.innerHTML = pwd + ': ' + guess_number;
  });
}
