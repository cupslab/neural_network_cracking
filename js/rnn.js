'use strict';

var UPPERCASE = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
var END_CHAR = '\n';
var INPUT_CLASS = '.nn-password';
var DISPLAY_CLASS = '.nn-display';
var NEURAL_NETWORK_SCRIPT = 'worker.js';
var NEURAL_NETWORK_INTERMEDIATE = 'js_information.json';

function NeuralNetworkClient(ctable, callback) {
  this.worker = new Worker(NEURAL_NETWORK_SCRIPT);
  this.worker.onmessage = this.onMessageTriggered.bind(this);
  this.callback = callback;
  this.ctable = ctable;
}

NeuralNetworkClient.prototype.onMessageTriggered = function(message) {
  this.callback(message);
};

NeuralNetworkClient.prototype.query = function(pwd) {
  this.worker.postMessage({
    inputData : this.ctable.encode_pwd(pwd)
  });
};

function CharacterTable(intermediate_info) {
  this.characters = intermediate_info.char_bag;
  this.pwd_len = intermediate_info.context_length;
  this.ctable_idx = {};
  this.rare_chars = {};
  this.backwards = intermediate_info.train_backwards;
  var rare_char_list = intermediate_info.rare_char_bag.join('')
  this.real_characters = intermediate_info.char_bag_real.join('');
  for (var i = 0; i < rare_char_list; i++) {
    this.real_characters = this.real_characters.replace(rare_char[i], '');
    this.rare_chars[rare_char_list[i]] = rare_char_list[0];
  }
  for (var i = 0; i < UPPERCASE.length; i++) {
    this.real_characters = this.real_characters.replace(UPPERCASE[i], '');
    this.rare_chars[UPPERCASE[i]] = UPPERCASE[i].toLowerCase();
  }
  for (var i = 0; i < this.real_characters.length; i++) {
    this.ctable_idx[this.real_characters[i]] = i;
  }
}

CharacterTable.prototype.encode_char = function(achar) {
  var answer = new Array(this.characters.length);
  var template_char = achar;
  if (achar in this.rare_chars) {
    template_char = this.rare_chars[achar];
  }
  var one_hot = this.ctable_idx[template_char];
  for (var i = 0; i < answer.length; i++) {
    if (i == one_hot) {
      answer[i] = 1;
    } else {
      answer[i] = 0;
    }
  }
  return answer;
};

CharacterTable.prototype.encode_pwd = function(pwd) {
  var query_text = pwd;
  if (pwd.length > this.pwd_len) {
    query_text = query_text.substring(pwd.length - this.pwd_len);
  }
  if (this.backwards) {
    query_text = query_text.split('').reverse().join('');
  }
  var chartable = new Array(this.pwd_len);
  for (var i = 0; i < query_text.length; i++) {
    chartable[i] = this.encode_char(query_text[i]);
  }
  for (var j = query_text.length; j < this.pwd_len; j++) {
    chartable[j] = this.encode_char(END_CHAR);
  }
  return chartable;
};

CharacterTable.prototype.decode_probs = function(prob_list) {
  var answer = {};
  for (var i = 0; i < this.real_characters.length; i++) {
    answer[this.real_characters[i]] = prob_list[i];
  }
  return answer;
}

function PwdInput(elem, output_elem, ctable) {
  this.elem = elem;
  this.ctable = ctable;
  this.client = new NeuralNetworkClient(
    this.ctable, this.onProbCallback.bind(this));
  this.output_elem = output_elem;
  var onchange = this.onChangeTriggered.bind(this)
  this.elem.change(onchange);
  this.elem.keydown(onchange);
  this.elem.keyup(onchange);
  this.elem.keypress(onchange);
  this.elem.blur(onchange);
}

PwdInput.prototype.onChangeTriggered = function(event) {
  this.client.query(this.elem.val());
};

PwdInput.prototype.onProbCallback = function(event) {
  var probs = this.ctable.decode_probs(event.data.prediction);
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
  $.getJSON(NEURAL_NETWORK_INTERMEDIATE, function(data) {
    console.log('Intermediate info loaded');
    var pwd_input = new PwdInput(
      $(INPUT_CLASS), $(DISPLAY_CLASS), new CharacterTable(data));
  });
}
