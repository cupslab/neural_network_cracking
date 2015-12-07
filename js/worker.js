'use strict';

importScripts('neocortex.min.js');

var ACTION_TOTAL_PROB = 'total_prob';
var ACTION_PREDICT_NEXT = 'predict_next';
var ACTION_RAW_PREDICT_NEXT = 'raw_predict_next';
var UPPERCASE = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
var END_CHAR = '\n';
var NEURAL_NETWORK_INTERMEDIATE = 'js_information.json';
var NEURAL_NETWORK_FILE = './js_model_small.json';

var nn = new NeuralNet({
  modelFilePath: NEURAL_NETWORK_FILE,
  arrayType: 'float32',
  useGPU: true
});

var ctable;
var onLoadMsgs = [];
onmessage = function(e) {
  onLoadMsgs.push(e);
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
  var answer = new Array(this.real_characters.length);
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
};

function predictNext(input_pwd) {
  return ctable.decode_probs(nn.predict(ctable.encode_pwd(input_pwd)))
}

function rawPredictNext(input_pwd) {
  return nn.predict(ctable.encode_pwd(input_pwd));
}

function totalProb(input_pwd, prefix) {
  var accum = 1;
  for (var i = 0; i < input_pwd.length; i++) {
    var values = ctable.decode_probs(nn.predict(
      ctable.encode_pwd(input_pwd.substring(0, i))));
    accum *= values[input_pwd[i]];
  }
  if (prefix) {
    var values = ctable.decode_probs(nn.predict(
      ctable.encode_pwd(input_pwd)));
    accum *= values[PASSWORD_END];
  }
  return accum;
}

function handleMsg(e) {
  if (e.data.action == ACTION_TOTAL_PROB) {
    self.postMessage({
      prediction : totalProb(e.data.inputData, e.data.prefix)
    });
  } else if (e.data.action == ACTION_PREDICT_NEXT) {
    self.postMessage({
      prediction : predictNext(e.data.inputData)
    });
  } else if (e.data.action == ACTION_RAW_PREDICT_NEXT) {
    self.postMessage({
      prediction : rawPredictNext(e.data.inputData)
    });
  } else {
    console.error('Unknown message action', e.data.action);
  }
}

var request = new XMLHttpRequest();
request.addEventListener('load', function() {
  console.log('Network loaded')
  ctable = new CharacterTable(JSON.parse(this.responseText));
  nn.init().then(function() {
    console.log('Worker ready for passwords!');
    onLoadMsgs.forEach(handleMsg);
    self.onmessage = handleMsg;
  }).catch(function(error) {
    console.error(error);
  });
});
request.open('GET', NEURAL_NETWORK_INTERMEDIATE);
request.send();
