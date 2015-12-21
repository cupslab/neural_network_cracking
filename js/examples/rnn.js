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
  this.client.predict_next(this.elem.val());
};

function init() {
  console.log('Loading scripts');
  var display_next = $(DISPLAY_CLASS);
  pwd_input = new PwdInput($(INPUT_CLASS), function(probs) {
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
    display_next.html(output_text);
  });
}
