var base64 = require('base64-js');
var md5 = require('md5');
var BITS_IN_BYTES = 8;

function BloomFilter(info) {
  this.info = this.decode(info);
  this.num = Object.keys(this.info).length;
  if (this.num == 0) {
    console.error('Gave empty list?!');
  }
  console.log('Loading bloom filter');
}

BloomFilter.prototype.decode = function(info) {
  var output = {};
  for (var i = 0; i < Object.keys(info).length; i++) {
    output[i] = base64.toByteArray(info[i]);
  }
  return output;
};

BloomFilter.prototype.hash_fn = function(pwd) {
  var digest = md5(pwd);
  var output = [];
  for (var i = 0; i < 4; i++) {
    output.push(parseInt(digest.substring(i * 8, (i + 1) * 8), 16));
  }
  // Only use the last 8 bytes
  return output[3];
};

BloomFilter.prototype.check_pwd_idx = function(hash_value, idx) {
  var data = this.info[idx];
  var bit_idx = hash_value % (data.length * BITS_IN_BYTES);
  var block_idx = Math.floor(bit_idx / BITS_IN_BYTES);
  return ((data[block_idx]) & (Math.pow(2, (bit_idx % BITS_IN_BYTES)))) != 0;
};

BloomFilter.prototype.check_pwd = function(pwd) {
  var hash_val = this.hash_fn(pwd);
  var one_works = false;
  for (var i = this.num - 1; i >= 0; i--) {
    if (!this.check_pwd_idx(hash_val, i)) {
      return one_works ? i + 1 : false;
    } else {
      one_works = true;
    }
  }
  return 0;
};

exports.BloomFilter = BloomFilter;
