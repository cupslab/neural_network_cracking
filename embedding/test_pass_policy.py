#!/usr/bin/env python

import unittest
import tempfile

import pass_policy

class PolicyTests(unittest.TestCase):
    def test_basic(self):
        policy = pass_policy.get_policy('basic')
        self.assertTrue(type(policy), pass_policy.BasicPolicy)
        self.assertTrue(policy.pwd_complies('asdf'))
        self.assertTrue(policy.pwd_complies('asdf' * 30))
        self.assertTrue(policy.pwd_complies(''))

    def test_one_uppercase(self):
        policy = pass_policy.get_policy('one_uppercase')
        self.assertTrue(type(policy), pass_policy.OneUppercasePolicy)
        self.assertFalse(policy.pwd_complies('asdf'))
        self.assertFalse(policy.pwd_complies('asdf' * 30))
        self.assertFalse(policy.pwd_complies(''))
        self.assertTrue(policy.pwd_complies('Asdf'))
        self.assertTrue(policy.pwd_complies('asDD'))

    def test_basic_long(self):
        policy = pass_policy.get_policy('basic_long')
        self.assertTrue(type(policy), pass_policy.RequiredLenPasswordPolicy)
        self.assertFalse(policy.pwd_complies('asdf'))
        self.assertFalse(policy.pwd_complies('asdfasd'))
        self.assertFalse(policy.pwd_complies(''))
        self.assertTrue(policy.pwd_complies('asdf' * 30))
        self.assertFalse(policy.pwd_complies('asdfasdf'))
        self.assertTrue(policy.pwd_complies('asdfasdfasdfasdf'))

    def test_complex(self):
        policy = pass_policy.get_policy('complex')
        self.assertTrue(type(policy), pass_policy.ComplexPasswordPolicy)
        self.assertTrue(policy.has_group('asdf', policy.lowercase))
        self.assertFalse(policy.has_group('1', policy.lowercase))
        self.assertTrue(policy.has_group('10', policy.digits))
        self.assertFalse(policy.has_group('a', policy.digits))
        self.assertTrue(policy.has_group('A', policy.uppercase))
        self.assertFalse(policy.has_group('a', policy.uppercase))
        self.assertTrue(policy.all_from_group('asdf0A', policy.non_symbols))
        self.assertFalse(policy.all_from_group('asdf*', policy.non_symbols))
        self.assertTrue(policy.passes_blacklist('asdf*'))
        self.assertFalse(policy.pwd_complies('asdf'))
        self.assertFalse(policy.pwd_complies('asdfasd'))
        self.assertFalse(policy.pwd_complies(''))
        self.assertFalse(policy.pwd_complies('asdf' * 30))
        self.assertFalse(policy.pwd_complies('asdfasdf'))
        self.assertFalse(policy.pwd_complies('asdfasdfasdfasdf'))
        self.assertFalse(policy.pwd_complies('1Aasdfasdfasdfasdf'))
        self.assertFalse(policy.pwd_complies('1Aa*'))
        self.assertFalse(policy.pwd_complies('1A*'))
        self.assertFalse(policy.pwd_complies('111*asdf'))
        self.assertTrue(policy.pwd_complies('1Aasdfasdfasdfasdf*'))
        self.assertTrue(policy.pwd_complies('1Aa*asdf'))
        self.assertTrue(policy.pwd_complies('999Apple*'))
        self.assertTrue(policy.pwd_complies('111*Asdf'))
        self.assertTrue(policy.pwd_complies('111*jjjJ'))
        with tempfile.NamedTemporaryFile(mode='w') as temp_bl:
            temp_bl.write('asdf\n')
            temp_bl.write('apple\n')
            temp_bl.flush()
            policy.load_blacklist(temp_bl.name)
        self.assertFalse(policy.pwd_complies('111*Asdf'))
        self.assertFalse(policy.pwd_complies('999Apple*'))
        self.assertTrue(policy.pwd_complies('111*jjjJ'))

    def test_complex_lowercase(self):
        policy = pass_policy.get_policy('complex_lowercase')
        self.assertTrue(type(policy), pass_policy.ComplexPasswordPolicyLowercase)
        self.assertTrue(policy.has_group('asdf', policy.lowercase))
        self.assertFalse(policy.has_group('1', policy.lowercase))
        self.assertTrue(policy.has_group('10', policy.digits))
        self.assertFalse(policy.has_group('a', policy.digits))
        self.assertTrue(policy.has_group('A', policy.uppercase))
        self.assertFalse(policy.has_group('a', policy.uppercase))
        self.assertTrue(policy.all_from_group('asdf0A', policy.non_symbols))
        self.assertFalse(policy.all_from_group('asdf*', policy.non_symbols))
        self.assertTrue(policy.passes_blacklist('asdf*'))
        self.assertFalse(policy.pwd_complies('asdf'))
        self.assertFalse(policy.pwd_complies('asdfasd'))
        self.assertFalse(policy.pwd_complies(''))
        self.assertFalse(policy.pwd_complies('asdf' * 30))
        self.assertFalse(policy.pwd_complies('asdfasdf'))
        self.assertFalse(policy.pwd_complies('asdfasdfasdfasdf'))
        self.assertFalse(policy.pwd_complies('1Aasdfasdfasdfasdf'))
        self.assertFalse(policy.pwd_complies('1Aa*'))
        self.assertFalse(policy.pwd_complies('1A*'))
        self.assertTrue(policy.pwd_complies('111*asdf'))
        self.assertTrue(policy.pwd_complies('1Aasdfasdfasdfasdf*'))
        self.assertTrue(policy.pwd_complies('1Aa*asdf'))
        self.assertTrue(policy.pwd_complies('999Apple*'))
        self.assertTrue(policy.pwd_complies('111*Asdf'))
        self.assertTrue(policy.pwd_complies('111*jjjJ'))
        with tempfile.NamedTemporaryFile(mode='w') as temp_bl:
            temp_bl.write('asdf\n')
            temp_bl.write('apple\n')
            temp_bl.flush()
            policy.load_blacklist(temp_bl.name)
        self.assertFalse(policy.pwd_complies('111*Asdf'))
        self.assertFalse(policy.pwd_complies('999Apple*'))
        self.assertTrue(policy.pwd_complies('111*jjjJ'))

if __name__ == '__main__':
    unittest.main()
