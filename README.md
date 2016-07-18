==========================================================
LIBSOL - A Library for Scalable Online Learning Algorithms
=========================================================


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


    + For users with Xcode

            $ cmake -G"Xcode" ..

    + For windows users

            $ cmake -G "Visual Studio 14 2015 Win64" ..

        Open the project ``libsol.sln``, Rebuild the `ALL_BUILD` project and then build the `INSTALL` project

        **Note**: Both 32-bit and 64-bit programs are ok to build and run. But if users want to use the python wrapper, it's required that the architectures of ``python`` and LIBSOL are the same, i.e., 64-bit ``python`` can only use 64-bit LIBSOL, 32-bit ``python`` can only use 32-bit LIBSOL.

3. The generated package will be copied to $LIBSOL/dist

4. Install python  wrapper. [optional]

        $ cd python && pip install -r requirements.txt


Quick Start
===========
Running LIBSOL without any arguments or with '--help' will produce a message which briefly explains the arguments.

We provide an example to show how to use LIBSOL and explain the details of how LIBSOL works.
The dataset we use will be `a1a` provided in the ``data`` folder.

The command for training wit default algorithm is as the following shows.

    $ libsol_train data/a1a
    training accuracy: 0.8125
    training time: 0.000 seconds
    model sparsity: 15.1260%

Users can use the python wrapper to do the same thing. But make sure that the
software is compiled with the same architecure of python (32-bit or 64-bit).

    $ python python/libsol_train.py data/a1a

The learned model can be saved to a file (``a1a.model`` for example) by:

    $ #using executable
    $ libsol_train data/a1a a1a.model
    $ #using python
    $ python python/libsol_train.py data/a1a a1a.model

By default, LIBSOL use ``OGD`` to learn a model. If users want to try another
algorithm (``AROW`` for example) and save to another file (``arow.model``):

    $ #using executable
    $ libsol_train -a arow data/a1a arow.model
    $ #using python
    $ python python/libsol_train.py -a arow data/a1a a1a.model

Each algorithm may have its own parameters. The following command changes the
default value of parameter ``r`` to ``2.0``:

    $ #using executable
    $ libsol_train -a arow --params r=2.0 data/a1a arow.model
    $ #using python
    $ python python/libsol_train.py --params r=2.0 -a arow data/a1a arow.model

The python wrapper also provides the cross validation ability. For example, if
users want to do a 5-fold GridSearch Cross Validation in the range [2^-5,2^-4,...,2^4, 2^5] for
parameter ``r`` of AROW, the command will be:

    $ python python/libsol_train.py -a arow --cv r=0.03125:2:32 -f 5 data/a1a arow.model
    cross validation parameters: [('r', 2.0)]

In some cases we want to finetune from a pretrained model,

    $ #using executable
    $ libsol_train -m arow.model data/a1a arow2.model
    $ #using python
    $ python python/libsol_train.py -m arow.model data/a1a arow2.model

We can test with the learned model:

    $ #using executable
    $ libsol_test arow.model data/a1a.t predict.txt
    $ #using python
    $ python python/libsol_test.py arow.model data/a1a.t
    test accuracy: 0.8437
    test time: 0.016 seconds

##Known Issues of Python Wrappers

- The wrappers are tested on Windows with Anaconda python distribution.

- On MacOS, the default python is not a framework build.  Seems matplotlib does not work properly. See [here](http://matplotlib.org/faq/virtualenv_faq.html) for more details. We recommend the Anaconda python distribution.

- On MacOS, if you met the 'Value Error: unknown locale: UTF-8' error, fix by:

        $ export LC_ALL=en_US.UTF-8
        $ export LANG=en_US.UTF-8


For  details, please check documentation of LIBSOL.

Comparison of Online Learning Algorithms
========================================
The ``experiment`` folder contains some scripts for a comparison of online
learning algorithms, as well as the comparison with another two toolboxes: VW
and LIBLINEAR. To quikly get a comparison on the small dataset ``a1a`` as
provided in the data folder:

    $ cd experiments
    $ python experiment.py --shufle 10 a1a ../data/a1a ../data/a1a.t

The script will conduct cross validation to select best parameters for each
algorithm. Then the script will shuffle the training 10 times. For each
shuffled data, the script will train and test for each algorithm. The final
output is the average of all results. And a final table report will be shown as follows.

	algorithm   train           train           test            test
	            accuracy        time(s)         accuracy        time(s)
	pa1         0.8017+/-0.0059 0.0132+/-0.0051 0.8112+/-0.0188 0.0283+/-0.0020
	pa2         0.7928+/-0.0080 0.0112+/-0.0021 0.8046+/-0.0242 0.0275+/-0.0033
	eccw        0.7945+/-0.0061 0.0123+/-0.0013 0.8030+/-0.0186 0.0279+/-0.0017
	arow        0.8186+/-0.0049 0.0142+/-0.0053 0.8409+/-0.0010 0.0274+/-0.0017
	perceptron  0.7697+/-0.0057 0.0140+/-0.0071 0.7895+/-0.0325 0.0269+/-0.0013
	ada-rda     0.8143+/-0.0047 0.0129+/-0.0011 0.8330+/-0.0076 0.0270+/-0.0008
	ada-fobos   0.8096+/-0.0076 0.0148+/-0.0065 0.8317+/-0.0105 0.0277+/-0.0020
	sop         0.7801+/-0.0067 0.0123+/-0.0017 0.7888+/-0.0196 0.0268+/-0.0015
	pa          0.7783+/-0.0065 0.0115+/-0.0013 0.7957+/-0.0253 0.0265+/-0.0031
	rda         0.7528+/-0.0007 0.0137+/-0.0056 0.7595+/-0.0000 0.0273+/-0.0028
	cw          0.7887+/-0.0072 0.0145+/-0.0067 0.7977+/-0.0220 0.0270+/-0.0023
	ogd         0.8102+/-0.0083 0.0118+/-0.0014 0.8336+/-0.0108 0.0280+/-0.0025
	alma2       0.8110+/-0.0062 0.0147+/-0.0070 0.8257+/-0.0120 0.0275+/-0.0020
	erda        0.8039+/-0.0059 0.0156+/-0.0062 0.8249+/-0.0137 0.0275+/-0.0015

There will also be three pdf figures displaying the update number, training error rate, and test error rate over model sparsity.

Users can also compare on the multi-class dataset
[``mnist``](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist) with the follow command (Note that we only shuffle the training data once in this example, so the standard deviation is zero):

    $ python experiment.py mnist ../data/mnist.scale ../data/mnist.scale.t

The output is:

	algorithm   train           train           test            test
	            accuracy        time(s)         accuracy        time(s)
	pa1         0.9859+/-0.0000 1.0040+/-0.0000 0.9939+/-0.0000 0.1630+/-0.0000
	pa2         0.9856+/-0.0000 1.0110+/-0.0000 0.9939+/-0.0000 0.1620+/-0.0000
	eccw        0.8892+/-0.0000 1.0290+/-0.0000 0.8865+/-0.0000 0.1780+/-0.0000
	arow        0.9911+/-0.0000 1.6110+/-0.0000 0.9949+/-0.0000 0.2800+/-0.0000
	perceptron  0.9836+/-0.0000 0.9560+/-0.0000 0.9912+/-0.0000 0.1630+/-0.0000
	ada-rda     0.9906+/-0.0000 1.0320+/-0.0000 0.9947+/-0.0000 0.1680+/-0.0000
	ada-fobos   0.9909+/-0.0000 1.0020+/-0.0000 0.9951+/-0.0000 0.1690+/-0.0000
	sop         0.9854+/-0.0000 1.0690+/-0.0000 0.9925+/-0.0000 0.1800+/-0.0000
	pa          0.9851+/-0.0000 2.8160+/-0.0000 0.9939+/-0.0000 0.4040+/-0.0000
	rda         0.9525+/-0.0000 1.0070+/-0.0000 0.9518+/-0.0000 0.1690+/-0.0000
	cw          0.9876+/-0.0000 1.0280+/-0.0000 0.9923+/-0.0000 0.1620+/-0.0000
	ogd         0.9885+/-0.0000 1.0150+/-0.0000 0.9927+/-0.0000 0.1630+/-0.0000
	alma2       0.9888+/-0.0000 1.0030+/-0.0000 0.9937+/-0.0000 0.1630+/-0.0000
	erda        0.9888+/-0.0000 1.0080+/-0.0000 0.9941+/-0.0000 0.1690+/-0.0000

The tables and figures in our paper description are obtained with the following
command:

    $ python experiment.py --shuffle 10 rcv1 ../data/rcv1_train ../data/rcv1_test

Additional Information
======================

For any questions and comments, please send your email to
chhoi@smu.edu.sg

Released date: 18 July, 2016.
