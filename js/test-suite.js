var TEST_CASES = 'test_cases.json';
var TEST_SET_DATA = 'test_data';

var test_cases;
var input_data;
var errors = 0;
var success = 1;

var EPSILON = 0.00001

function assert(value, message) {
  if (value) {
    success += 1;
  } else {
    console.error(message, 'ERROR!');
    errors += 1;
  }
}

function assertEqual(valueA, valueB, message) {
  var value = valueA == valueB;
  if (value) {
    success += 1;
  } else {
    console.error(message, 'ERROR', valueA, valueB);
    errors += 1;
  }
}

function assertAlmostEqual(valueA, valueB, message) {
  var value = Math.abs(valueA - valueB);
  if (value < EPSILON) {
    success += 1;
  } else {
    console.error(message, 'ERROR', valueA, valueB);
    errors += 1;
  }
}

function testCase1(done) {
  var key = 'test_case1_conditional_prob';
  var i = 0;
  var fn_cb = function(ret_val) {
    var expected = test_cases[key][i];
    assertEqual(expected[0].length, ret_val.length, 'Outputs correct length');
    for (var j = 0; j < ret_val.length; j++) {
      assertAlmostEqual(ret_val[j], expected[0][j],
                        'Outputs correct prediction for ' + j);
    }

    i += 1;
    if (i < input_data.length) {
      console.log('testing', input_data[i][0], '...');
      client.raw_predict_next(input_data[i][0]);
    } else {
      done();
    }
  };
  client.callback = fn_cb;
  client.raw_predict_next(input_data[i][0]);
}

var client = new NeuralNetworkClient(null);

function doTests() {
  testCase1(function() {
    console.log('done errors:', errors, 'successes:', success);
  });
}

function setupRunTestCases(tc) {
  test_cases = tc;
  input_data = tc[TEST_SET_DATA];
  doTests();
}

function init() {
  $.getJSON(TEST_CASES, setupRunTestCases);
  console.log('loading...');
}
