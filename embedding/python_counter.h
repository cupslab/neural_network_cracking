#include <Python.h>

PyObject* countCharacters(
    const char* filename,
    const char* alphabet,
    const char* policy_name,
    const char* end_of_pass_char,
    bool separate_begin);
