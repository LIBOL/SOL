#! /usr/bin/env python
#################################################################################
#     File Name           :     setup.py
#     Created By          :     yuewu
#     Description         :      
#################################################################################

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "lsol",
        sources=["lsol.pyx"],
        libraries=["lsol"],
        include_dirs=[np.get_include(), "/home/yuewu/work/libsol/include"],
        library_dirs=["/home/yuewu/work/libsol/dist/bin"],
        extra_compile_args = ["-DHAS_NUMPY_DEV"]
        )
]

setup(name="lsol", ext_modules=cythonize(ext_modules))
