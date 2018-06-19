#include "policy.h"

#include <regex>
#include <string>
#include <cassert>



#include <iostream>

namespace PassPolicy {

using string = std::string;



AlphabetLookupTable::AlphabetLookupTable() :
  fast_to_idx_(),
  slow_to_idx_() {}

void AlphabetLookupTable::setIndex(uint32_t character, int value) {
  if (character < LOOKUP_SIZE) {
    fast_to_idx_[character] = value;
  } else {
    slow_to_idx_[character] = value;
  }
}

const std::string& AlphabetLookupTable::getAlphabet() const {
  return alphabet_;
}

void AlphabetLookupTable::init(
    const std::string& alphabet,
    const std::vector<int>& values,
    int default_value) {
  default_value_ = default_value;
  alphabet_ = alphabet;

  fast_to_idx_.reserve(LOOKUP_SIZE);
  for (size_t j = 0; j < LOOKUP_SIZE; ++j) {
    fast_to_idx_.push_back(default_value_);
  }

  auto iter = alphabet.begin();
  auto end = alphabet.end();
  size_t i = 0;
  while (iter != end) {
    setIndex(utf8::next(iter, end), values[i++]);
  }
  assert(values.size() == i);
}

void AlphabetLookupTable::init(const std::string& alphabet, int default_value) {
  std::vector<int> values;
  values.reserve(alphabet.size());
  auto iter = alphabet.begin();
  auto end = alphabet.end();
  int i = 0;
  while (iter != end) {
    utf8::next(iter, end);
    values.push_back(i++);
  }
  init(alphabet, values, default_value);
}




GrowableCharacterCounter::GrowableCharacterCounter() :
  fast_char_to_idx_(),
  char_to_idx_() {
  for (size_t i = 0; i < LOOKUP_SIZE; ++i) {
    fast_char_to_idx_.push_back(0);
  }
}

std::vector<uint32_t> GrowableCharacterCounter::getKeys() const {
  std::vector<uint32_t> answer;
  for (size_t i = 0; i < LOOKUP_SIZE; ++i) {
    if (fast_char_to_idx_[i] != 0) {
      answer.push_back(i);
    }
  }
  auto end = char_to_idx_.end();
  for (auto beg = char_to_idx_.begin(); beg != end; ++beg) {
    answer.push_back(beg->first);
  }
  return answer;
}

int64_t GrowableCharacterCounter::countAt(uint32_t key) const {
  if (key < LOOKUP_SIZE) {
    return fast_char_to_idx_[key];
  } else {
    auto iter = char_to_idx_.find(key);
    if (iter != char_to_idx_.end()) {
      return iter->second;
    } else {
      return 0;
    }
  }
}


CharacterCounter::CharacterCounter() : alpha_(), counts_() {}

void CharacterCounter::init(const std::string& alphabet) {
  alpha_.init(alphabet);
  int num_elems = utf8::distance(alphabet.begin(), alphabet.end());
  counts_.reserve(num_elems);
  for (int i = 0; i < num_elems; ++i) {
    counts_.push_back(0);
  }
}

size_t CharacterCounter::size() const {
  return counts_.size();
}

void CharacterCounter::reset() {
  std::fill(counts_.begin(), counts_.end(), 0);
}

std::vector<uint32_t> CharacterCounter::getKeys() const {
  std::vector<uint32_t> answer;
  const std::string& alpha = alpha_.getAlphabet();
  answer.reserve(alpha.size());
  auto beg = alpha.begin();
  auto end = alpha.end();
  while (beg != end) {
    answer.push_back(utf8::next(beg, end));
  }
  return answer;
}

int64_t CharacterCounter::countAt(uint32_t key) const {
  return counts_[alpha_.indexFor(key)];
}

const std::vector<int64_t>& CharacterCounter::getCounts() const {
  return counts_;
}





PasswordPolicy::PasswordPolicy() {}


inline size_t utf8Size(const string& value) {
  return utf8::distance(value.begin(), value.end());
}

inline bool passesSizeRequirement(const string& value, size_t required) {
  if (value.size() < required) {
    // Quick check: if we have fewer bytes than are required, then we can't meet
    // the requirements
    return false;
  }
  return utf8Size(value) >= required;
}

class BasicPolicy : public PasswordPolicy {
public:
  bool passwordComplies(const string& value) override {
    return true;
  }

  bool passwordComplies(const std::string& value, size_t size) override {
    return true;
  }
};


class LengthRequirementPolicy : public PasswordPolicy {
public:
  explicit LengthRequirementPolicy(size_t required) : required_(required) {}

  bool passwordComplies(const string& value) override {
    return passesSizeRequirement(value, required_);
  }

  bool passwordComplies(const std::string& value, size_t size) override {
    return size >= required_;
  }

private:
  size_t required_;
};


class BaseComplexPolicy : public PasswordPolicy {
public:
  static constexpr size_t LOWER = 1;
  static constexpr size_t UPPER = 2;
  static constexpr size_t DIGIT = 4;
  static constexpr size_t SYMBOL = 8;

  static constexpr size_t MAX_FLAG = 16;
  static constexpr size_t MAX_FLAG_BITS = 4;

  static constexpr size_t NUM_LETTERS = 26;
  static constexpr size_t NUM_DIGITS = 10;

  BaseComplexPolicy() {
    std::vector<int> values;
    std::string alphabet (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
    values.reserve(alphabet.size());
    for (size_t i = 0; i < NUM_LETTERS; ++i) {
      values.push_back(LOWER);
    }
    for (size_t i = 0; i < NUM_LETTERS; ++i) {
      values.push_back(UPPER);
    }
    for (size_t i = 0; i < NUM_DIGITS; ++i) {
      values.push_back(DIGIT);
    }
    flags_.init(alphabet, values, SYMBOL);


    for (size_t j = 0; j < MAX_FLAG; ++j) {
      flag_vals_[j] = countNumberOfClasses(j);
    }

    for (size_t j = 0; j < MAX_FLAG; ++j) {
      flag_vals_ign_case_[j] = countNumberOfClassesIgnoringCase(j);
    }
  }

  unsigned int flagMask(const string& value) {
    unsigned int flag = 0;
    auto iter = value.begin();
    auto end = value.end();
    while (iter != end) {
      flag |= static_cast<unsigned int>(flags_.indexFor(utf8::next(iter, end)));
    }
    return flag;
  }

  size_t numberOfClasses(const string& value) {
    unsigned int mask = flagMask(value);
    assert(mask < MAX_FLAG);
    return flag_vals_[mask];
  }

  size_t numberOfClassesIgnoringCase(const string& value) {
    return flag_vals_ign_case_[flagMask(value)];
  }

  inline bool hasRequiredClasses(const string& value, size_t required) {
    return hasRequiredClasses(value, required, flag_vals_);
  }

  inline bool hasRequiredClassesIgnoreCase(
      const string& value, size_t required) {
    return hasRequiredClasses(value, required, flag_vals_ign_case_);
  }

private:
  bool hasRequiredClasses(const string& value, size_t required, size_t* vals) {
    unsigned int flag = 0;
    auto iter = value.begin();
    auto end = value.end();
    while (iter != end) {
      flag |= static_cast<unsigned int>(flags_.indexFor(utf8::next(iter, end)));
      if (vals[flag] >= required) {
        return true;
      }
    }
    return false;
  }

  size_t countNumberOfClasses(unsigned int flag) {
    return
      ((flag & DIGIT) ? 1 : 0) +
      ((flag & SYMBOL) ? 1 : 0) +
      ((flag & LOWER) ? 1 : 0) +
      ((flag & UPPER) ? 1 : 0);
  }

  size_t countNumberOfClassesIgnoringCase(size_t flag) {
    return
      ((flag & DIGIT) ? 1 : 0) +
      ((flag & SYMBOL) ? 1 : 0) +
      ((flag & (LOWER | UPPER)) ? 1 : 0);
  }

  AlphabetLookupTable flags_;

  size_t flag_vals_[MAX_FLAG];
  size_t flag_vals_ign_case_[MAX_FLAG];
};


class ComplexPolicy : public BaseComplexPolicy {
public:
  ComplexPolicy(size_t required_length, size_t num_classes) :
    required_length_(required_length),
    num_classes_(num_classes) {}

  bool passwordComplies(const string& value) override {
    return passesSizeRequirement(value, required_length_) &&
      hasRequiredClasses(value, num_classes_);
  }

  bool passwordComplies(const std::string& value, size_t size) override {
    return size >= required_length_ && hasRequiredClasses(value, num_classes_);
  }


private:
  size_t required_length_;
  size_t num_classes_;
};


class ComplexLowercasePolicy : public BaseComplexPolicy {
public:
  ComplexLowercasePolicy(size_t required_length, size_t num_classes) :
    required_length_(required_length),
    num_classes_(num_classes) {}

  bool passwordComplies(const string& value) override {
    return passesSizeRequirement(value, required_length_) &&
      hasRequiredClassesIgnoreCase(value, num_classes_);
  }

  bool passwordComplies(const std::string& value, size_t size) override {
    return size >= required_length_ &&
      hasRequiredClassesIgnoreCase(value, num_classes_);
  }

private:
  size_t required_length_;
  size_t num_classes_;
};


std::unique_ptr<PasswordPolicy> PasswordPolicy::fromString(string name) {
  PasswordPolicy* value = nullptr;
  // Should match pass_policy.py
  if (name == "basic") {
    value = new BasicPolicy();
  } else if (name == "1class8") {
    value = new LengthRequirementPolicy(8);
  } else if (name == "basic_long") {
    value = new LengthRequirementPolicy(16);
  } else if (name == "complex_lowercase") {
    value = new ComplexLowercasePolicy(8, 3);
  } else if (name == "complex") {
    value = new ComplexPolicy(8, 4);
  } else if (name == "complex_long") {
    value = new ComplexPolicy(16, 4);
  } else if (name == "complex_long_lowercase") {
    value = new ComplexLowercasePolicy(16, 3);
  } else if (name == "semi_complex") {
    value = new ComplexPolicy(12, 3);
  } else if (name == "semi_complex_lowercase") {
    value = new ComplexLowercasePolicy(12, 2);
  } else if (name == "3class12") {
    value = new ComplexPolicy(12, 3);
  } else if (name == "3class12_all_lowercase") {
    value = new ComplexLowercasePolicy(12, 3);
  } else {
    std::regex xclassy ("([0-9]+)class([0-9]+)");
    std::smatch match;
    if (std::regex_match(name, match, xclassy)) {
      int class_num = std::stoi(match[1]);
      int char_num = std::stoi(match[2]);
      if (class_num == 1) {
        value = new LengthRequirementPolicy(char_num);
      } else {
        value = new ComplexPolicy(char_num, class_num);
      }
    }
  }
  return std::unique_ptr<PasswordPolicy>(value);
}


}
