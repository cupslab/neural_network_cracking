#ifndef POLICY_HPP
#define POLICY_HPP

#include <stdint.h>
#include <memory>
#include <vector>
#include <unordered_map>

#include "third_party/utf8.h"

#include <iostream>


namespace PassPolicy {

static constexpr size_t LOOKUP_SIZE = 256;


class AlphabetLookupTable {
public:
  AlphabetLookupTable();

  void init(const std::string& alphabet, int default_value = -1);
  void init(const std::string& alphabet,
            const std::vector<int>& values,
            int default_value = -1);

  inline int indexFor(uint32_t character) const {
    if (character < LOOKUP_SIZE) {
      return fast_to_idx_[character];
    }

    auto iter = slow_to_idx_.find(character);
    if (iter != slow_to_idx_.end()) {
      return iter->second;
    } else {
      return default_value_;
    }
  }

  inline bool allCharsInTable(const std::string& target) const {
    return allCharsInTable(target, nullptr);
  }

  inline bool allCharsInTable(const std::string& target, size_t* size) const {
    auto iter = target.begin();
    auto end = target.end();
    int num = 0;
    while (iter != end) {
      uint32_t code_point;
      try {
        code_point = utf8::next(iter, end);
      } catch (utf8::not_enough_room&) {
        return false;
      } catch (utf8::invalid_utf8&) {
        return false;
      } catch (utf8::invalid_code_point&) {
        return false;
      }
      if (indexFor(code_point) < 0) {
        return false;
      }
      num += 1;
    }
    if (size != nullptr) {
      *size = num;
    }
    return true;
  }


  const std::string& getAlphabet() const;

private:
  void setIndex(uint32_t character, int value);

  int default_value_;
  std::vector<int> fast_to_idx_;
  std::unordered_map<uint32_t, int> slow_to_idx_;
  std::string alphabet_;
};


class GrowableCharacterCounter {
public:

  GrowableCharacterCounter();

  inline void accum(uint32_t character) {
    if (character < LOOKUP_SIZE) {
      fast_char_to_idx_[character] += 1;
    } else {
      auto iter = char_to_idx_.find(character);
      if (iter != char_to_idx_.end()) {
        iter->second += 1;
      } else {
        char_to_idx_[character] = 1;
      }
    }
  }

  inline void accum(const std::string& str) {
    auto beg = str.begin();
    auto end = str.end();
    while (beg != end) {
      accum(utf8::next(beg, end));
    }
  }

  std::vector<uint32_t> getKeys() const;

  int64_t countAt(uint32_t key) const;

private:
  std::vector<int64_t> fast_char_to_idx_;
  std::unordered_map<uint32_t, int64_t> char_to_idx_;
};


class CharacterCounter {
public:
  CharacterCounter();

  const std::vector<int64_t>& getCounts() const;

  void reset();

  inline void accum(const std::string& value) {
    auto iter = value.begin();
    auto end = value.end();
    while (iter != end) {
      increment(utf8::next(iter, end));
    }
  }


  inline void increment(uint32_t code_point) {
    counts_[alpha_.indexFor(code_point)] += 1;
  }

  std::vector<uint32_t> getKeys() const;
  int64_t countAt(uint32_t key) const;

  void init(const std::string& alphabet);

  size_t size() const;


private:
  AlphabetLookupTable alpha_;
  std::vector<int64_t> counts_;
};


class PasswordPolicy {
public:
  PasswordPolicy();

  virtual bool passwordComplies(const std::string& value) = 0;

  // Fast check for if we already have the size of the string in utf8 code
  // points
  virtual bool passwordComplies(const std::string& value, size_t size) = 0;

  static std::unique_ptr<PasswordPolicy> fromString(std::string name);
};


}


#endif /* POLICY_HPP */
