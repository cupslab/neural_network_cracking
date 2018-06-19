#include "python_counter.h"
#include "policy.h"

#include "third_party/utf8.h"

#include <fstream>
#include <iostream>


using namespace PassPolicy;


PyObject* fromCounter(const CharacterCounter& counter) {
  auto keys = counter.getKeys();
  size_t keys_size = keys.size();
  PyObject* output = PyDict_New();
  for (size_t i = 0; i < keys_size; ++i) {
    std::string key;
    uint32_t code_point = keys[i];
    utf8::append(code_point, std::back_inserter(key));
    PyObject* pykey = PyUnicode_FromStringAndSize(key.data(), key.size());
    PyObject* value = PyLong_FromLong(counter.countAt(code_point));
    if (PyDict_SetItem(output, pykey, value) != 0) {
      return Py_None;
    }
  }
  return output;
}


PyObject* countCharacters(
    const char* filename,
    const char* alphabet,
    const char* policy_name,
    const char* end_of_pass_char,
    bool separate_begin) {
  std::unique_ptr<PasswordPolicy> policy =
    PasswordPolicy::fromString(policy_name);

  if (!policy) {
    std::cerr << "No policy " << policy_name << "\n";
    return Py_None;
  }

  CharacterCounter begin;
  CharacterCounter normal;
  AlphabetLookupTable table;

  std::string end_of_pass_str (end_of_pass_char);
  std::string alpha_str (alphabet);
  begin.init(alpha_str + end_of_pass_str);
  normal.init(alpha_str + end_of_pass_str);
  table.init(alpha_str);

  std::ifstream is (filename);
  if (!is.good()) {
    std::cerr << "Cannot open " << filename << "\n";
    return Py_None;
  }

  std::string line;

  auto end_iter = end_of_pass_str.begin();
  auto end_end = end_of_pass_str.end();
  if (end_iter == end_end) {
    std::cerr << "Empty end of password character\n";
    return Py_None;
  }

  uint32_t end_code_point = utf8::next(end_iter, end_end);
  while (std::getline(is, line)) {
    size_t size;
    if (table.allCharsInTable(line, &size)) {
      if (policy->passwordComplies(line, size)) {
        auto iter = line.begin();
        const auto end = line.end();
        if (iter != end) {
          uint32_t first_code_point = utf8::next(iter, end);
          if (separate_begin) {
            begin.increment(first_code_point);
          } else {
            normal.increment(first_code_point);
          }
          while (iter != end) {
            normal.increment(utf8::next(iter, end));
          }
        }

        if (separate_begin) {
          begin.increment(end_code_point);
        } else {
          normal.increment(end_code_point);
        }
      }
    }
  }

  PyObject* result = PyDict_New();
  PyDict_SetItemString(result, "begin", fromCounter(begin));
  PyDict_SetItemString(result, "normal", fromCounter(normal));
  return result;
}
