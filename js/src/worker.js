'use strict';

var USE_BLOOM_FILTER = false;
var TO_LOWERCASE = false;

// Change these
var NEURAL_NETWORK_INTERMEDIATE =
      'basic_3M.info_and_guess_numbers.json';
var NEURAL_NETWORK_FILE =
      'basic_3M.weight_arch.quantized.fixed_point1000.zigzag.nospace.json';
var ZIGZAG = true;
var SCALE_FACTOR = 1;



// For testing quantizing
// For no quantization
// var NEURAL_NETWORK_FILE = 'basic_3M.weight_arch.json';



// For complex
// var NEURAL_NETWORK_FILE =
//       'complex_3M.weight_arch.quantized2digits.fixedpoint.json';
// var NEURAL_NETWORK_INTERMEDIATE = 'complex_3M_info_and_gn.json';
// var ZIGZAG = false;
// var SCALE_FACTOR = 100;



var jscache = require('js-cache');
var bs = require('binarysearch');
var bloom_filter = require('./bloom_filter');
import NeuralNet from 'neocortex-rnn';

var ACTION_TOTAL_PROB = 'total_prob';
var ACTION_PREDICT_NEXT = 'predict_next';
var ACTION_RAW_PREDICT_NEXT = 'raw_predict_next';
var ACTION_GUESS_NUMBER = 'guess_number';
var UPPERCASE = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
var END_CHAR = '\n';
var CACHE_SIZE = 100;


var bf;
var nn;
var ctable;
var cached_table;
var guess_numbers;
var gn_cache = new jscache.Cache(CACHE_SIZE);
var onLoadMsgs = [];

self.onmessage = function(e) {
  onLoadMsgs.push(e);
};

function CharacterTable(intermediate_info) {
  this.characters = intermediate_info.char_bag;
  this.pwd_len = intermediate_info.context_length;
  this.min_len = intermediate_info.min_len || 0;
  this.ctable_idx = {};
  this.rare_chars = {};
  this.rare_chars_opposite = {};
  var add_to_rco = (function(from, to) {
    if (!(from in this.rare_chars_opposite)) {
      this.rare_chars_opposite[from] = [];
    }
    this.rare_chars_opposite[from].push(to);
  }).bind(this);
  this.backwards = intermediate_info.train_backwards;
  var rare_char_list = intermediate_info.rare_char_bag.join('');
  this.real_characters = intermediate_info.char_bag_real.join('');
  for (var i = 1; i < rare_char_list.length; i++) {
    if (UPPERCASE.indexOf(rare_char_list[i]) != -1) {
      continue;
    }
    this.real_characters = this.real_characters.replace(rare_char_list[i], '');
    this.rare_chars[rare_char_list[i]] = rare_char_list[0];
    add_to_rco(rare_char_list[0], rare_char_list[i]);
  }
  add_to_rco(rare_char_list[0], rare_char_list[0]);
  this.rare_chars[rare_char_list[0]] = rare_char_list[0];
  for (var j = 0; j < UPPERCASE.length; j++) {
    this.real_characters = this.real_characters.replace(UPPERCASE[j], '');
    var lowercase = UPPERCASE[j].toLowerCase();
    this.rare_chars[UPPERCASE[j]] = lowercase;
    this.rare_chars[lowercase] = lowercase;
    add_to_rco(lowercase, lowercase);
    add_to_rco(lowercase, UPPERCASE[j]);
  }
  for (var k = 0; k < this.real_characters.length; k++) {
    this.ctable_idx[this.real_characters[k]] = k;
  }
  this.rare_character_calc = this.calc_cache(
    intermediate_info.character_frequencies);
  this.beginning_rare_character_calc = this.calc_cache(
    intermediate_info.beginning_character_frequencies);
}

CharacterTable.prototype.calc_cache = function(freqs) {
  var answer = {};
  for (var fromKey in this.rare_chars_opposite) {
    var toKeys = this.rare_chars_opposite[fromKey];
    answer[fromKey] = {};
    var sum_prob = 0;
    for (var i = 0; i < toKeys.length; i++) {
      sum_prob += freqs[toKeys[i]];
    }
    for (var j = 0; j < toKeys.length; j++) {
      answer[fromKey][toKeys[j]] = freqs[toKeys[j]] / sum_prob;
    }
  }
  return answer;
};

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
    var sum = answer.reduce(function(prev, cur) { return prev + cur; });
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

CharacterTable.prototype.probability_of_char = function(
  prob_list, next_char, beginning) {
  var value = this.decode_probs(prob_list);
  var answer;
  if (next_char in value) {
    answer = value[next_char];
  } else {
    answer = value[this.rare_chars[next_char]];
  }
  var prev = answer;
  if (next_char in this.rare_chars) {
    var template_char = this.rare_chars[next_char];
    if (!beginning) {
      answer *= this.rare_character_calc[template_char][next_char];
    } else {
      answer *= this.beginning_rare_character_calc[template_char][next_char];
    }
  }
  return answer;
};

function ProbCacher(size, nn, table) {
  this.cached_probabilities = new jscache.Cache(size);
  this.nn = nn;
  this.table = table;
}

ProbCacher.prototype.probability_of_char = function(
  prefix, next_char, beginning) {
  var cached_value = this.cached_probabilities.get(prefix);
  if (cached_value) {
    return this.table.probability_of_char(cached_value, next_char, beginning);
  } else {
    var all_probs = this.table.cond_prob(prefix, this.nn);
    var answer = this.table.probability_of_char(
      all_probs, next_char, beginning);
    this.cached_probabilities.set(prefix, all_probs);
    return answer;
  }
};

function rawPredictNext(input_pwd) {
  return ctable.cond_prob(input_pwd, nn);
}

function totalProb(input_pwd, prefix) {
  var accum = 1;
  for (var i = 0; i < input_pwd.length; i++) {
    accum *= cached_table.probability_of_char(
      input_pwd.substring(0, i), input_pwd[i],
      i == 0);
  }
  if (!prefix) {
    accum *= cached_table.probability_of_char(
      input_pwd, END_CHAR, input_pwd == '');
  }
  return accum;
}

function lookupGuessNumber(input_pwd) {
  var cached_value = gn_cache.get(input_pwd);
  if (cached_value) {
    return cached_value;
  }
  var prob = totalProb(input_pwd, false);
  if (prob == 0) {
    gn_cache.set(input_pwd, -1);
    return -1;
  }
  var bs_search = bs.closest(guess_numbers, prob, function(value, find) {
    if (value[0] > find) return 1;
    else if (value[0] < find) return -1;
    else return 0;
  });
  if (bs_search == 0) {
    gn_cache.set(input_pwd, Infinity);
    return Infinity;
  }
  var guess_number_answer;
  if (bs_search + 1 > guess_numbers.length) {
    guess_number_answer = guess_numbers[guess_numbers.length - 1];
  } else {
    guess_number_answer = guess_numbers[bs_search + 1];
  }
  var answer = Math.round(guess_number_answer[1]);
  gn_cache.set(input_pwd, answer);
  return answer;
}

function lookupGuessNumberWithBloomFilter(input_pwd) {
  var nn_guess = lookupGuessNumber(
    TO_LOWERCASE ? input_pwd.toLowerCase() : input_pwd) / SCALE_FACTOR;
  if (bf !== undefined) {
    var bf_guess = bf.check_pwd(input_pwd);
    if (bf_guess !== false) {
      return Math.min(nn_guess, Math.pow(10, bf_guess));
    }
  }
  return nn_guess;
};

function predictNext(input_pwd) {
  return ctable.decode_probs(ctable.cond_prob(input_pwd, nn));
}

function handleMsg(e) {
  var message;
  var pwd = e.data.inputData;
  if (e.data.action == ACTION_TOTAL_PROB) {
    message = {
      prediction : totalProb(e.data.inputData, e.data.prefix),
      password : pwd
    };
  } else if (e.data.action == ACTION_GUESS_NUMBER) {
    message = {
      prediction : lookupGuessNumberWithBloomFilter(e.data.inputData),
      password : pwd
    };
  } else if (e.data.action == ACTION_PREDICT_NEXT) {
    message = {
      prediction : predictNext(e.data.inputData),
      password : pwd
    };
  } else if (e.data.action == ACTION_RAW_PREDICT_NEXT) {
    message = {
      prediction : rawPredictNext(e.data.inputData),
      password : pwd
    };
  } else {
    console.error('Unknown message action', e.data.action);
  }
  self.postMessage(message);
}

var request = new XMLHttpRequest();
request.addEventListener('load', function() {
  var info = JSON.parse(this.responseText);
  ctable = new CharacterTable(info);
  guess_numbers = info['guessing_table'];
  if (info['bloom_filter'] && USE_BLOOM_FILTER) {
    bf = new bloom_filter.BloomFilter(info['bloom_filter']);
  }
  nn = new NeuralNet({
    modelFilePath: NEURAL_NETWORK_FILE,
    arrayType: 'float32',
    useGPU: true,
    scaleFactor: info['fixed_point_scale'],
    msgPackFmt: false,
    zigzagEncoding: ZIGZAG
  });
  console.log(nn, info);
  cached_table = new ProbCacher(CACHE_SIZE, nn, ctable);
  nn.init(function() {
    onLoadMsgs.forEach(handleMsg);
    self.onmessage = handleMsg;
  });
});
request.open('GET', NEURAL_NETWORK_INTERMEDIATE);
request.send();
