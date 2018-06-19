
#include "python_counter.h"

extern "C" {

static PyObject*
countCharactersPy(PyObject* self, PyObject* args) {
  const char* filename;
  const char* alphabet;
  const char* policy_name;
  const char* end_of_pass_char;
  int separate_begin;
  if (!PyArg_ParseTuple(
      args, "ssssi",
      &filename,
      &alphabet,
      &policy_name,
      &end_of_pass_char,
      &separate_begin)) {
    return NULL;
  }

  return countCharacters(
      filename, alphabet, policy_name, end_of_pass_char, separate_begin != 0);
}

static struct PyMethodDef methods[] = {
  {"count_chars", (PyCFunction) countCharactersPy, METH_VARARGS,
   "Count characters in password file."},
  {NULL, NULL, 0, NULL}         // Sentinel
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "count_characters",
  NULL,
  -1,
  methods
};

PyMODINIT_FUNC
PyInit_count_characters(void) {
  return PyModule_Create(&module);
}

}
