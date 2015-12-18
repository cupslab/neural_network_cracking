var TEST_SET_PWDS_FILE = 'test_set_pwds.json';
var TEST_SET_PWDS;
var i = 0;
var output_text = "";

var client = new NeuralNetworkClient(callback);

function output(data, pwd) {
  output_text += pwd + '\t' + data + '\n';
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
    $('#output').text(output_text);
    console.log('done');
  }
}

function init() {
  $.getJSON(TEST_SET_PWDS_FILE, function(data){
    TEST_SET_PWDS = data;
    callback();
  });
}
