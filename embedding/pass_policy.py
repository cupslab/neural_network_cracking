import string


class BasePasswordPolicy(object):
    def pwd_complies(self, pwd):
        raise NotImplementedError()

class BasicPolicy(BasePasswordPolicy):
    def pwd_complies(self, pwd):
        return True

class RequiredLenPasswordPolicy(BasePasswordPolicy):
    def __init__(self, length):
        self._req_len = length

    def pwd_complies(self, pwd):
        return len(pwd) >= self._req_len

class ComplexPasswordPolicy(BasePasswordPolicy):
    digits = set(string.digits)
    uppercase = set(string.ascii_uppercase)
    lowercase = set(string.ascii_lowercase)
    upper_and_lowercase = set(string.ascii_uppercase + string.ascii_lowercase)
    non_symbols = set(
      string.digits + string.ascii_uppercase + string.ascii_lowercase)

    def __init__(self, required_length=8):
        self.blacklist = set()
        self.required_length = required_length

    def load_blacklist(self, fname):
        with open(fname, 'r') as blacklist:
            for line in blacklist:
                self.blacklist.add(line.strip('\n'))

    def has_group_symbols(self, pwd):
        return not self.all_from_group(pwd, self.non_symbols)

    def has_group(self, pwd, group):
        return any(map(lambda c: c in group, pwd))

    def all_from_group(self, pwd, group):
        return all(map(lambda c: c in group, pwd))

    def passes_blacklist(self, pwd):
        return (''.join(filter(
          lambda c: c in self.upper_and_lowercase, pwd)).lower()
            not in self.blacklist)

    def pwd_complies(self, pwd):
        if len(pwd) < self.required_length:
            return False
        if not self.has_group(pwd, self.digits):
            return False
        if not self.has_group(pwd, self.uppercase):
            return False
        if not self.has_group(pwd, self.lowercase):
            return False
        if self.all_from_group(pwd, self.non_symbols):
            return False
        return self.passes_blacklist(pwd)

class ComplexPasswordPolicyLowercase(ComplexPasswordPolicy):
    def pwd_complies(self, pwd):
        if len(pwd) < self.required_length:
            return False
        if not self.has_group(pwd, self.digits):
            return False
        if not self.has_group(pwd, self.upper_and_lowercase):
            return False
        if self.all_from_group(pwd, self.non_symbols):
            return False
        return self.passes_blacklist(pwd)

class OneUppercasePolicy(ComplexPasswordPolicy):
    def pwd_complies(self, pwd):
        if len(pwd) < self.required_length:
            return False
        if not self.has_group(pwd, self.uppercase):
            return False
        return self.passes_blacklist(pwd)

class SemiComplexPolicyLowercase(ComplexPasswordPolicy):
    def pwd_complies(self, pwd):
        count = 0
        if len(pwd) < self.required_length:
            return False
        if self.has_group(pwd, self.digits):
            count += 1
        if self.has_group(pwd, self.upper_and_lowercase):
            count += 1
        if self.has_group_symbols(pwd):
            count += 1
        return self.passes_blacklist(pwd) and count >= 2

class SemiComplexPolicy(ComplexPasswordPolicy):
    def pwd_complies(self, pwd):
        count = 0
        if len(pwd) < self.required_length:
            return False
        if self.has_group(pwd, self.digits):
            count += 1
        if self.has_group(pwd, self.uppercase):
            count += 1
        if self.has_group(pwd, self.lowercase):
            count += 1
        if self.has_group_symbols(pwd):
            count += 1
        return self.passes_blacklist(pwd) and count >= 3


# Should match expand_operation.cc
policy_list = {
  'complex' : ComplexPasswordPolicy(),
  'basic' : BasicPolicy(),
  '1class8' : RequiredLenPasswordPolicy(8),
  'basic_long' : RequiredLenPasswordPolicy(16),
  'complex_lowercase' : ComplexPasswordPolicyLowercase(),
  'complex_long' : ComplexPasswordPolicy(16),
  'complex_long_lowercase' : ComplexPasswordPolicyLowercase(16),
  'semi_complex' : SemiComplexPolicy(12),
  'semi_complex_lowercase' : SemiComplexPolicyLowercase(12),
  '3class12' : SemiComplexPolicy(12),
  '2class12_all_lowercase' : SemiComplexPolicyLowercase(12),
  'one_uppercase' : OneUppercasePolicy(3)
}


class PasswordValidator(object):
    def __init__(self, alphabet):
        self._alpha = set(alphabet)

    def is_valid(self, pwd):
        return all([c in self._alpha for c in pwd])


class PasswordFilterer(object):
    def __init__(self, alphabet, policy):
        self._validator = PasswordValidator(alphabet)
        self._policy = get_policy(policy)
        self.number_removed = 0
        self.number_passed = 0

    def __call__(self, pwd):
        result = self._validator.is_valid(pwd) and self._policy.pwd_complies(pwd)
        self.number_removed += (0 if result else 1)
        self.number_passed += (1 if result else 0)
        return result

    def reset(self):
        self.number_removed = 0
        self.number_passed = 0


def policies():
    return list(policy_list.keys())

def get_policy(name):
    return policy_list[name]
