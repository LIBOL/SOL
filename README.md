=====================================================
LIBSOL - A Large Scale Sparse Online Learning Library
====================================================


About LIBSOL
===========================================================================
LIBSOL is an open-source library for scalable online learning with high-dimensional data. The library provides a family of regular and sparse online learning algorithms for large-scale binary and multi-class classification tasks with high efficiency, scalability, portability, and extensibility. We provide easy-to-use command-line tools, python wrappers and library calls for users and developers, and comprehensive documents for both beginners and advanced users. LIBSOL is not only a machine learning toolbox, but also a comprehensive experimental platform for online learning research. Experiments demonstrate that LIBSOL is highly efficient and scalable for large-scale learning with high-dimensional data.

Specifically, LIBSOL consists of a family of:
+ First Order Online feature selection algorithms:
    - Perceptron: The Perceptron Algorithm(Rosenblatt, 1958)
    - OGD: Online Gradient Descent(Zinkevich, 2003)
    - PA: Online Passive Aggressive Algorithms(Crammer et al., 2006)
    - ALMA: Approximate Large Margin Algorithm(Gentile, 2002)
    - RDA: Regularized Dual Averaging(Xiao, 2010)
+ Second order online learning algorithms:
    - SOP: Second-Order Perceptron(Cesa-Bianchi et al., 2005)
    - CW: Confidence Weighted Learning(Dredze et al., 2008)
    - ECCW: Exactly Convex Confidence Weighted Learning(Crammer et al., 2008)
    - AROW: Adaptive Regularized Online Learning(Crammer et al., 2009)
    - Ada-FOBOS: Adaptive Regularized Online Learning(Crammer et al., 2009)
    - Ada-RDA: Adaptive Regularized Dual Averaging(Crammer et al., 2009)

+ First order Sparse online learning algorithms:
    - STG: sparse online learning via truncated graidient (Langford et al., 2009);
    - FOBOS-L1: l1 Regularized Forward backward splitting (Duchi et al., 2009);
    - RDA-L1: Mixed l1/l2^2 Regularized Dual averaging(Xiao, 2010);
    - ERDA-L1: Enhanced l1/l2^2 Regularized Dual averaging(Xiao, 2010);

+ Second order sparse online learning algorithms as follows
    - Ada-FOBOS-L1: Ada-FOBOS with l1 regularization
    - Ada-RDA-L1: Ada-RDA with l1 regularization

This document briefly explains the usage of LIBSOL. A more detailed manual can be
found from documentation of LIBSOL.

To get started, please read the ``Quick Start'' section first.

Table of Contents
=================
- Installation
- Quick Start
- Additional Information

Installation
======================
LIBSOL features a very simple installation procedure. The project is managed by Cmake. There exists a `CMakeLists.txt` in the root directory.

##Prerequisites

+ CMake  2.8.12 or higher
+ Git
+ C++11 Compiler:
    - g++4.8.2 or higher on Unix/Linux/MinGW/Cygwin
    - Visual Studio 2013 or higher on Windows
+ Python2.7 (required if you want to use the python wrappers)

##Getting the code

The latest version of LIBSOL is always available via 'github' by invoking one
of the following:

    ## For the traditional ssh-based Git interaction:
    $ git clone git://github.com/LIBOL/LIBSOL.git

    ## For HTTP-based Git interaction
    $ git clone https://github.com/LIBOL/LIBSOL.git

##Compiling

1. Make a folder to store project files:

        $ cd LIBSOL && mkdir build && cd build

2. Generate and build the project files

    + For Unix/Linux/MacOS users

            $ cmake ..
            $ make -j
            $ make install


    + For users with Eclipse

            $ cmake -G"Eclipse CDT4 - Unix Makefiles" ..

    + For windows users

            $ cmake -G "Visual Studio 14 2015 Win64" ..

        Open the project ``libsol.sln``, Rebuild the `ALL_BUILD` project and then build the `INSTALL` project

        **Note**: Both 32-bit and 64-bit programs are ok to build and run. But if users want to use the python wrapper, it's required that the architectures of ``python`` and LIBSOL are the same, i.e., 64-bit ``python`` can only use 64-bit LIBSOL, 32-bit ``python`` can only use 32-bit LIBSOL.

3. The generated package will be copied to $LIBSOL/dist

Quick Start
===========
Running LIBSOL without any arguments or with '--help' will produce a message which briefly explains the arguments.

We provide an example to show how to use LIBSOL and explain the details of how LIBSOL works.
The dataset we use will be `a1a` provided in the ``data`` folder.

The command for training wit default algorithm is as the following shows.

    $ lsol_train data/a1a
    training accuracy: 0.8125
    training time: 0.000 seconds
    model sparsity: 15.1260%

Users can use the python wrapper to do the same thing. 

    $ python python/lsol_train.py data/a1a

The learned model can be saved to a file (``a1a.model`` for example) by:

    $ #using executable
    $ lsol_train data/a1a a1a.model
    $ #using python
    $ python python/lsol_train.py data/a1a a1a.model

By default, LIBSOL use ``OGD`` to learn a model. If users want to try another
algorithm (``AROW`` for example) and save to another file (``arow.model``):

    $ #using executable
    $ lsol_train -a arow data/a1a arow.model
    $ #using python
    $ python python/lsol_train.py -a arow data/a1a a1a.model

Each algorithm may have its own parameters. The following command changes the
default value of parameter ``r`` to ``2.0``:

    $ #using executable
    $ lsol_train -a arow --params r=2.0 data/a1a arow.model
    $ #using python
    $ python python/lsol_train.py --params r=2.0 -a arow data/a1a arow.model

The python wrapper also provides the cross validation ability. For example, if
users want to do a 5-fold GridSearch Cross Validation in the range [2^-5,2^-4,...,2^4, 2^5] for
parameter ``r`` of AROW, the command will be:

    $ python python/lsol_train.py -a arow --cv r=0.03125:2:32 -f 5 data/a1a arow.model
    cross validation parameters: [('r', 2.0)]

In some cases we want to finetune from a pretrained model,

    $ #using executable
    $ lsol_train -m arow.model data/a1a arow2.model
    $ #using python
    $ python python/lsol_train.py -m arow.model data/a1a arow2.model

We can test with the learned model:

    $ #using executable
    $ lsol_test arow.model data/a1a.t predict.txt
    $ #using python
    $ python python/lsol_test.py arow.model data/a1a.t
    test accuracy: 0.8437
    test time: 0.016 seconds

For  details, please check documentation of LIBSOL.

Additional Information
======================

For any questions and comments, please send your email to
chhoi@smu.edu.sg

Released date: 11 July, 2016.
