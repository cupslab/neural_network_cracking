var INPUT_CLASS = '.nn-password';
var DISPLAY_CLASS = '.nn-display';
var pwd_input;

function PwdInput(elem, output_elem) {
  this.elem = elem;
  this.client = new NeuralNetworkClient(this.onProbCallback.bind(this));
  this.output_elem = output_elem;
  var onchange = this.onChangeTriggered.bind(this)
  this.elem.change(onchange);
  this.elem.keydown(onchange);
  this.elem.keyup(onchange);
  this.elem.keypress(onchange);
  this.elem.blur(onchange);
}

PwdInput.prototype.onChangeTriggered = function(event) {
  this.client.predict_next(this.elem.val());
};

PwdInput.prototype.onProbCallback = function(probs) {
  var output_text = '';
  var keys = Object.keys(probs);
  var keys_sorted = keys.map(function(k) {
    return [probs[k], k];
  }).sort(function(a, b) {
    return b[0] - a[0];
  });
  for (var i = 0; i < keys_sorted.length; i++) {
    output_text += keys_sorted[i][1] + ': ' + keys_sorted[i][0] + '<br/>';
  }
  this.output_elem.html(output_text);
};

function init() {
  console.log('Loading scripts');
  pwd_input = new PwdInput($(INPUT_CLASS), $(DISPLAY_CLASS));
}
