#! /usr/bin/env python
#################################################################################
#     File Name           :     setup.py
#     Created By          :     yuewu
#     Description         :
#################################################################################

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

from Cython.Build import cythonize

import sys
import os

try:
    from pypandoc import convert

    def read_md(fpath):
        return convert(fpath, 'rst')

except ImportError:
    print("warning: pypandoc module not found, DONOT convert Markdown to RST")

    def read_md(fpath):
        with open(fpath, 'r') as fp:
            return fp.read()

sys.path.append("python")


def get_source_files(root_dir):
    src_files = []
    for pathname in os.listdir(root_dir):
        path = os.path.join(root_dir, pathname)
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1]
            if ext in ['.cc', '.cpp', '.c']:
                src_files.append(path)
        elif os.path.isdir(path):
            src_files = src_files + get_source_files(path)
    return src_files
def get_include_dirs():
    import numpy as np
    return [np.get_include(), "include", "external"]

if os.name == 'nt':
    extra_flags = ['/wd4251','/wd4275', '/EHsc','-DSOL_EMBED_PACKAGE']
    dependencies = []
else:
    extra_flags = ['-std=c++11','-pthread']
    dependencies = [
        "numpy          >= 1.7.0",
        "scipy          >= 0.13.0",
        "scikit-learn   >= 0.18.1",
        "matplotlib     >= 1.5.1"
    ]


ext_modules = [
    Extension(
        "pysol",
        sources=["python/pysol.pyx"] + get_source_files('src/sol') +
        get_source_files('external/json'),
        language='c++',
        include_dirs=get_include_dirs(),
        extra_compile_args=['-DHAS_NUMPY_DEV', '-DUSE_STD_THREAD'] + extra_flags)
]


setup(
    name='sol',
    version='1.1.0',
    description='Library for Scalable Online Learning',
    #long_description=read_md('README.md'),
    author='Yue Wu, Steven C.H. Hoi',
    author_email='yuewu@outlook.com',
    maintainer='Yue Wu',
    maintainer_email='yuewu@outlook.com',
    url='http://sol.stevenhoi.org',
    license='Apache 2.0',
    keywords='Scalable Online Learning',
    packages=['sol'],
    package_dir={'sol': 'python'},
    entry_points = {
        'console_scripts':[
            'sol_train=sol.sol_train:main',
            'sol_test=sol.sol_test:main',
            ],
        },
    ext_modules=cythonize(ext_modules),
    install_requires=dependencies
    )
