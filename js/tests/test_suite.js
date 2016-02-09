var TEST_CASES = 'generated_js_test_cases_basic3M.json';
var TEST_SET_DATA = 'test_data';

var test_cases;
var input_data;
var errors = 0;
var success = 1;

var EPSILON = 0.000001;

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
    console.error(message, 'ERROR', valueA, '!=', valueB);
    errors += 1;
  }
}

function assertAlmostEqual(valueA, valueB, message) {
  var value = Math.abs(valueA - valueB);
  if (value <= EPSILON || (!isFinite(valueA)) && !isFinite(valueB)) {
    success += 1;
  } else {
    console.error(message, 'ERROR', valueA, '!=', valueB);
    errors += 1;
  }
}

function testCase1(done) {
  var key = 'test_case1_conditional_prob';
  var i = 0;
  var msg = 'testing conditional probability';
  var fn_cb = function(ret_val) {
    var expected = test_cases[key][i];
    assertEqual(expected.length, ret_val.length, 'Outputs correct length');
    for (var j = 0; j < ret_val.length; j++) {
      assertAlmostEqual(ret_val[j], expected[j],
                        'Outputs correct prediction for ' + j);
    }
    i += 1;
    if (i < input_data.length) {
      console.log(msg, input_data[i][0], '...');
      client.raw_predict_next(input_data[i][0]);
    } else {
      done();
    }
  };
  console.log(msg, input_data[i][0], '...');
  client.callback = fn_cb;
  client.raw_predict_next(input_data[i][0]);
}

function testCase2(done) {
  var key = 'test_case2_total_prob_template_prefix';
  var i = 0;
  var msg = 'testing prefix probability';
  var fn_cb = function(ret_val) {
    var expected = test_cases[key][i];
    assertAlmostEqual(Math.log(ret_val), Math.log(expected[1]),
                      'Outputs correct prefix prediction on log scale');
    i += 1;
    if (i < input_data.length) {
      console.log(msg, input_data[i][0], '...');
      client.query(input_data[i][0], true);
    } else {
      done();
    }
  };
  client.callback = fn_cb;
  console.log(msg, input_data[i][0], '...');
  client.query(input_data[i][0], true);
}

function testCase3(done) {
  var key = 'test_case3_total_prob_template_noprefix';
  var i = 0;
  var msg = 'testing total probability';
  var fn_cb = function(ret_val) {
    var expected = test_cases[key][i];
    assertAlmostEqual(Math.log(ret_val), Math.log(expected[1]),
                      'Outputs correct prefix prediction on log scale');
    i += 1;
    if (i < input_data.length) {
      console.log(msg, input_data[i][0], '...');
      client.query(input_data[i][0]);
    } else {
      done();
    }
  };
  client.callback = fn_cb;
  console.log(msg, input_data[i][0], '...');
  client.query(input_data[i][0]);
}

var client = new NeuralNetworkClient(null);

function doTests() {
  console.log('running test case 1');
  testCase1(function() {
    // Log probabilities should be compared to lower accuracy
    EPSILON = 0.0001;
    console.log('running test case 2');
    testCase2(function() {
      console.log('running test case 3');
      testCase3(function() {
        console.log('done errors:', errors, 'successes:', success);
      });
    });
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
