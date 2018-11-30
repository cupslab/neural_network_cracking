#include "policy.h"

// This tells Catch to provide a main() - only do this in one cpp file
#define CATCH_CONFIG_MAIN
#include "third_party/catch.hpp"

#include "third_party/utf8.h"

using namespace PassPolicy;

TEST_CASE("complex policy works") {
  auto policy = PasswordPolicy::fromString("complex");
  REQUIRE(!policy->passwordComplies("asdfasdf"));
  REQUIRE(!policy->passwordComplies("Asdfasdf"));
  REQUIRE(!policy->passwordComplies("Asdfasd9"));
  REQUIRE(policy->passwordComplies("Asdfas9!"));
}

TEST_CASE("3class12 policy works") {
  auto policy = PasswordPolicy::fromString("3class12");
  REQUIRE(!policy->passwordComplies("asdfasdf1234"));
  REQUIRE(!policy->passwordComplies("Asdfasdfasdf"));
  REQUIRE(!policy->passwordComplies("asdfasdf!@#$"));
  REQUIRE(policy->passwordComplies("Asdfasd9asdf"));
  REQUIRE(policy->passwordComplies("Asdfas9!asdf"));
}

TEST_CASE("1class12 policy works utf8") {
  auto policy = PasswordPolicy::fromString("1class12");
  REQUIRE(!policy->passwordComplies("asdfasdfasà"));
  REQUIRE( policy->passwordComplies("asdfasdfasaa"));
}

TEST_CASE("xclassy policy works") {
  auto policy = PasswordPolicy::fromString("2class10");
  REQUIRE(!policy->passwordComplies("asdfasdfas"));
  REQUIRE(policy->passwordComplies("Asdfasdfas"));
  REQUIRE(policy->passwordComplies("Asdfasd9as"));
  REQUIRE(!policy->passwordComplies("Asdfas9!"));
}

TEST_CASE("basic policy works") {
  auto policy = PasswordPolicy::fromString("basic");
  REQUIRE(policy->passwordComplies("asdfasdfas"));
  REQUIRE(policy->passwordComplies("Asdfasdfas"));
  REQUIRE(policy->passwordComplies("Asdfasd9as"));
  REQUIRE(policy->passwordComplies("Asdfas9!"));
}

TEST_CASE("policy doesn't crash on utf8") {
  auto policy = PasswordPolicy::fromString("complex");
  REQUIRE(!policy->passwordComplies(u8"rôtter"));
  REQUIRE(!policy->passwordComplies(u8"rèfjkdfj"));
  REQUIRE(!policy->passwordComplies(u8"rèfjkdfj"));
  REQUIRE(policy->passwordComplies(u8"Rôtter99"));
}

TEST_CASE("alphabet works ascii indexFor") {
  AlphabetLookupTable alpha;
  alpha.init("abcdefghijklmnopqrstuvwxyz");
  REQUIRE(alpha.indexFor('a') == 0);
  REQUIRE(alpha.indexFor('b') == 1);
  REQUIRE(alpha.indexFor('z') == 25);
  REQUIRE(alpha.indexFor('9') == -1);
}

TEST_CASE("alphabet works nonascii indexFor") {
  AlphabetLookupTable alpha;
  alpha.init(u8"abcdefghijklmnopqrstuvwxyz");
  REQUIRE(alpha.indexFor('a') == 0);
  REQUIRE(alpha.indexFor('b') == 1);
  REQUIRE(alpha.indexFor('z') == 25);

  std::string ahat (u8"à");
  auto beg = ahat.begin();
  uint32_t a_hat_codepoint = utf8::next(beg, ahat.end());
  REQUIRE(alpha.indexFor(a_hat_codepoint) == -1);

  AlphabetLookupTable alpha2;
  alpha2.init(u8"abcdefghijklmnopqrstuvwxyzà");
  REQUIRE(alpha2.indexFor(a_hat_codepoint) == 26);
}

TEST_CASE("alphabet works allCharsInTable") {
  AlphabetLookupTable alpha;
  alpha.init("abcdefghijklmnopqrstuvwxyz");
  REQUIRE(alpha.allCharsInTable(u8"rotter"));
  REQUIRE(!alpha.allCharsInTable(u8"rotter0"));
  REQUIRE(!alpha.allCharsInTable(u8"rotteruö"));
  REQUIRE(!alpha.allCharsInTable(u8"rôtter"));
}

TEST_CASE("alphabet works utf8 alpha allCharsInTable") {
  AlphabetLookupTable alpha;
  alpha.init(u8"abcdefghijklmnopqrstuvwxyzà");
  REQUIRE(alpha.allCharsInTable(u8"rotter"));
  REQUIRE(!alpha.allCharsInTable(u8"rotter0"));
  REQUIRE(!alpha.allCharsInTable(u8"rotteruö"));
  REQUIRE(!alpha.allCharsInTable(u8"rôtter"));
  REQUIRE(alpha.allCharsInTable(u8"rotterà"));

  // à is utf code point c3 a0. Check that an partial unicode point with those
  // values does not pass.
  REQUIRE(!alpha.allCharsInTable("rotter\xC3"));
}


TEST_CASE("character counter works") {
  CharacterCounter counter;
  counter.init("abcde");
  counter.accum("abcdee");
  const auto& counts = counter.getCounts();
  REQUIRE(counts[0] == 1);
  REQUIRE(counts[1] == 1);
  REQUIRE(counts[2] == 1);
  REQUIRE(counts[3] == 1);
  REQUIRE(counts[4] == 2);
  counter.reset();
  REQUIRE(counts[0] == 0);
  REQUIRE(counts[1] == 0);
  REQUIRE(counts[2] == 0);
  REQUIRE(counts[3] == 0);
  REQUIRE(counts[4] == 0);
  REQUIRE(counter.size() == 5);
}

TEST_CASE("character counter works utf8") {
  CharacterCounter counter;
  counter.init(u8"abcdeà");
  counter.accum("abcdeeà");
  const auto& counts = counter.getCounts();
  REQUIRE(counts[0] == 1);
  REQUIRE(counts[1] == 1);
  REQUIRE(counts[2] == 1);
  REQUIRE(counts[3] == 1);
  REQUIRE(counts[4] == 2);
  REQUIRE(counts[5] == 1);
  REQUIRE(counter.size() == 6);
}
