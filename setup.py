#!/usr/bin/env python3

from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('custom_functions.pyx',compiler_directives={'language_level' : "3"}))
