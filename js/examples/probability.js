var INPUT_CLASS = '.nn-password';
var DISPLAY_CLASS = '.nn-display';
var pwd_input;

function PwdInput(elem, callback) {
  this.elem = elem;
  this.client = new NeuralNetworkClient(callback);
  var onchange = this.onChangeTriggered.bind(this);
  this.elem.change(onchange);
  this.elem.keydown(onchange);
  this.elem.keyup(onchange);
  this.elem.keypress(onchange);
  this.elem.blur(onchange);
}

PwdInput.prototype.onChangeTriggered = function(event) {
  this.client.query_guess_number(this.elem.val());
};

function init() {
  console.log('Loading scripts');
  var display_next = $(DISPLAY_CLASS);
  pwd_input = new PwdInput($(INPUT_CLASS), function(guess_number, pwd) {
    display_next.html(pwd + ': ' + guess_number);
  });
}
