#include "policy.h"

#include <string.h>

#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <algorithm>

#include "third_party/utf8.h"

using namespace PassPolicy;


char usage[] =
  " [-alphabet alphabet] [-policy policy] [-output] [-quiet] [-input file] [-help]\n"
  "Reads passwords one line at a time.\n"
  "-policy policy\tOnly allow passwords that match the given policy.\n"
  "-output\tWrite matching passwords to stdout.\n"
  "-quiet\tDon't print statistics\n"
  "-input\tInput path. If not given, will read from stdin.\n"
  "-alphabet\tProvide expected alphabet. If not given, will provide default that includes typical characters on an english keyboard. If ':none:' is given, then alphabet will not be enforced.\n"
  "-help\tPrint this meassage and exit\n";


struct OutputStats {
  uint64_t number_passwords = 0;
  uint64_t number_valid_passwords = 0;
  uint64_t number_passwords_match_policy = 0;
  uint64_t number_training_instances = 0;
  std::vector<std::tuple<std::string, int64_t>> character_count;
};


class AlphabetAllowAll {
public:
  inline bool allCharsInTable(const std::string& line, size_t* size) const {
    *size = static_cast<size_t>(utf8::distance(line.begin(), line.end()));
    return true;
  }

  inline bool allCharsInTable(const std::string& line) const {
    return true;
  }
};

template <typename KeyIgnored>
bool sort_counts(std::tuple<KeyIgnored, int64_t> a,
                 std::tuple<KeyIgnored, int64_t> b) {
  return std::get<1>(a) > std::get<1>(b);
}


template <typename CharacterCounterT>
std::vector<std::tuple<std::string, int64_t>> sortedCounts(
    CharacterCounterT& counter) {
  std::vector<uint32_t> keys = counter.getKeys();
  const size_t keys_size = keys.size();
  std::vector<std::tuple<std::string, int64_t>> keys_counts;
  keys_counts.reserve(keys_size);
  for (size_t i = 0; i < keys_size; ++i) {
    std::string key;
    uint32_t code_point = keys[i];
    utf8::append(code_point, std::back_inserter(key));
    keys_counts.push_back(std::make_tuple(key, counter.countAt(code_point)));
  }

  std::sort(keys_counts.begin(), keys_counts.end(), sort_counts<std::string>);
  return keys_counts;
}


template <bool should_output,
          typename CharacterCounterT,
          typename AlphabetEnforcerT>
void DoHandleFile(std::ostream& output,
                  std::istream& istream,
                  PasswordPolicy& policy,
                  const AlphabetEnforcerT& alpha,
                  CharacterCounterT& counter,
                  OutputStats* stats) {
  std::string line;
  while (std::getline(istream, line)) {
    stats->number_passwords += 1;

    size_t size;
    if (alpha.allCharsInTable(line, &size)) {
      stats->number_valid_passwords += 1;
      if (policy.passwordComplies(line, size)) {
        stats->number_passwords_match_policy += 1;
        stats->number_training_instances += size + 1;
        counter.accum(line);
        if (should_output) {
          output << line << "\n";
        }
      }
    }
  }

  stats->character_count = std::move(sortedCounts(counter));
}


void HandleFile(bool should_output,
                bool should_enforce_alpha,
                std::istream& istream,
                const char* alphabet,
                PasswordPolicy& policy,
                OutputStats* out_stats) {
  if (should_output) {
    if (should_enforce_alpha) {
      CharacterCounter counter;
      AlphabetLookupTable alpha;
      alpha.init(alphabet);
      counter.init(alphabet);
      DoHandleFile<true>(
          std::cout,
          istream,
          policy,
          alpha,
          counter,
          out_stats);
    } else {
      GrowableCharacterCounter counter;
      DoHandleFile<true>(
          std::cout,
          istream,
          policy,
          AlphabetAllowAll(),
          counter,
          out_stats);
    }
  } else {
    if (should_enforce_alpha) {
      CharacterCounter counter;
      AlphabetLookupTable alpha;
      alpha.init(alphabet);
      counter.init(alphabet);
      DoHandleFile<false>(
          std::cout,
          istream,
          policy,
          alpha,
          counter,
          out_stats);
    } else {
      GrowableCharacterCounter counter;
      DoHandleFile<false>(
          std::cout,
          istream,
          policy,
          AlphabetAllowAll(),
          counter,
          out_stats);
    }
  }
}


int main(int argc, char** argv) {
  bool print_usage_and_exit = false;
  bool output = false;
  bool quiet = false;
  char* input_path = nullptr;
  const char* policy_name = nullptr;
  const char* alphabet = nullptr;
  for (int i = 1; i < argc; ++i) {
    char* arg = argv[i];
    if (strcmp(arg, "-policy") == 0) {
      if (i + 1 >= argc) {
        fputs("Error: -policy flag needs argument\n", stderr);
        print_usage_and_exit = true;
      } else {
        i += 1;
        policy_name = argv[i];
      }
    } else if (strcmp(arg, "-input") == 0) {
      if (i + 1 >= argc) {
        fputs("Error: -input flag needs argument\n", stderr);
        print_usage_and_exit = true;
      } else {
        i += 1;
        input_path = argv[i];
      }
    } else if (strcmp(arg, "-alphabet") == 0) {
      if (i + 1 >= argc) {
        fputs("Error: -alphabet flag needs argument\n", stderr);
        print_usage_and_exit = true;
      } else {
        i += 1;
        alphabet = argv[i];
      }
    } else if (strcmp(arg, "-output") == 0) {
      output = true;
    } else if (strcmp(arg, "-help") == 0) {
      print_usage_and_exit = true;
    } else if (strcmp(arg, "-quiet") == 0) {
      quiet = true;
    } else {
      std::cerr << "Error: unrecognized argument " << arg << "\n";
      print_usage_and_exit = true;
    }
  }

  if (print_usage_and_exit) {
    if (argc >= 1) {
      std::cerr << argv[0];
    } else {
      std::cerr << "main_policy";
    }
    std::cerr << usage;
    return 1;
  }

  if (policy_name == nullptr) {
    policy_name = "basic";
  }

  bool enforce_alphabet = true;
  if (alphabet == nullptr) {
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`~@#$%^&*()-_=+[]{}\\|;:'\",<>./?!";
  } else if (strcmp(alphabet, ":none:") == 0) {
    enforce_alphabet = false;
    alphabet = "";
  }

  std::unique_ptr<PasswordPolicy> policy =
    PasswordPolicy::fromString(policy_name);

  if (!policy) {
    std::cerr << "Error: policy " << policy_name << " not recognized\n";
    return 1;
  }

  OutputStats stats;

  std::cout.sync_with_stdio(false);
  if (input_path) {
    std::ifstream input_file (input_path);
    HandleFile(output, enforce_alphabet, input_file, alphabet, *policy, &stats);
  } else {
    std::cin.sync_with_stdio(false);
    HandleFile(output, enforce_alphabet, std::cin, alphabet, *policy, &stats);
  }

  if (!quiet) {
    std::cerr
      << "Number of passwords: " << stats.number_passwords << "\n"
      << "Number of valid passwords: " << stats.number_valid_passwords << "\n"
      << "Number of passwords that match policy: "
      << stats.number_passwords_match_policy
      << "\n"
      << "Number training instances: "
      << stats.number_training_instances << "\n";

    std::cerr << "Character counts:\n";
    const std::vector<std::tuple<std::string, int64_t>>& values =
      stats.character_count;

    for (size_t i = 0; i < values.size(); ++i) {
      std::cerr << "'" << std::get<0>(values[i])
                << "'\t" << std::get<1>(values[i]) << "\n";
    }
  }

  return 0;
}
