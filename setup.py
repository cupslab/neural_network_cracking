from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    cmdclass = { 'build_ext': build_ext },
    version = '0.0.1',
    ext_modules = [Extension("generator", ["pwd_guess_ctypes.pyx"],
                             include_dirs = [numpy.get_include()])]
)
