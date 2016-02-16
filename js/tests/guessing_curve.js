var TEST_SET_PWDS_FILE = 'test_set_pwds.json';
var TEST_SET_PWDS;
var i = 0;
var output_string_builder = [];
var client;

function output(data, pwd) {
  output_string_builder.push(pwd + '\t' + data);
}

function callback(prob, pwd) {
  if (pwd) {
    output(prob, pwd);
  }
  if (i < TEST_SET_PWDS.length) {
    console.log(TEST_SET_PWDS[i]);
    client.query_guess_number(TEST_SET_PWDS[i]);
    i += 1;
    $('#output').html('working... ' + i + '/' + TEST_SET_PWDS.length);
  } else {
    $('#output').text(output_string_builder.join('\n'));
    console.log('done');
  }
}

client = factory(callback);

function init() {
  // For use with NN
  // $.getJSON(TEST_SET_PWDS_FILE, function(data){
  //   TEST_SET_PWDS = data;
  //   callback();
  // });

  // For use in zxcvbn
  TEST_SET_PWDS = TEST_PASSWORDS;
  // TEST_SET_PWDS.sort();         // to take advantage of cache effects for
  //                               // quicker tests
  callback();
}
