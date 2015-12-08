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
  this.min_len = intermediate_info.min_len || 0;
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
  query_text = query_text + (
    Array(this.pwd_len - query_text.length + 1).join(END_CHAR));
  if (this.backwards) {
    query_text = query_text.split('').reverse().join('');
  }
  var chartable = new Array(this.pwd_len);
  for (var i = 0; i < query_text.length; i++) {
    chartable[i] = this.encode_char(query_text[i]);
  }
  return chartable;
};

CharacterTable.prototype.cond_prob = function(pwd, nn) {
  var answer = nn.predict(this.encode_pwd(pwd));
  if (pwd.length < this.min_len) {
    answer[this.ctable_idx[END_CHAR]] = 0;
    var sum = answer.reduce( (prev, cur) => prev + cur );
    for (var j = 0; j < answer.length; j++) {
      answer[j] = answer[j] / sum;
    }
  }
  return answer;
};

CharacterTable.prototype.decode_probs = function(prob_list) {
  var answer = {};
  for (var i = 0; i < this.real_characters.length; i++) {
    answer[this.real_characters[i]] = prob_list[i];
  }
  return answer;
};

CharacterTable.prototype.probability_of_char = function(prob_list, next_char) {
  var value = ctable.decode_probs(prob_list);
  if (next_char in value) {
    return value[next_char];
  } else {
    return value[this.rare_chars[next_char]];
  }
};

function predictNext(input_pwd) {
  return ctable.decode_probs(ctable.cond_prob(input_pwd, nn));
}

function rawPredictNext(input_pwd) {
  return ctable.cond_prob(input_pwd, nn);
}

function totalProb(input_pwd, prefix) {
  var accum = 1;
  for (var i = 0; i < input_pwd.length; i++) {
    accum *= ctable.probability_of_char(
      ctable.cond_prob(input_pwd.substring(0, i), nn),
      input_pwd[i]);
  }
  if (prefix) {
    accum *= ctable.probability_of_char(ctable.cond_prob(input_pwd, nn),
                                        END_CHAR);
  }
  return accum;
}

function handleMsg(e) {
  var message;
  if (e.data.action == ACTION_TOTAL_PROB) {
    message = {
      prediction : totalProb(e.data.inputData, e.data.prefix)
    };
  } else if (e.data.action == ACTION_PREDICT_NEXT) {
    message = {
      prediction : predictNext(e.data.inputData)
    };
  } else if (e.data.action == ACTION_RAW_PREDICT_NEXT) {
    message = {
      prediction : rawPredictNext(e.data.inputData)
    };
  } else {
    console.error('Unknown message action', e.data.action);
  }
  self.postMessage(message);
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
