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
    - g++(>4.8.2) or clang++(>3.3) on Unix/Linux/MinGW/Cygwin
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

        Open the project ``LIBSOL.sln``, Rebuild the `ALL_BUILD` project and then build the `INSTALL` project


3. The generated package will be copied to $LIBSOL/dist

4. Install python  wrapper. [optional]

        $ cd python && pip install -r requirements.txt

	**Known Issues of Python Wrappers**:
	
	- The wrappers are tested on Windows with Anaconda python distribution.
	
	- On MacOS, the default python is not a framework build.  Seems matplotlib does not work properly. See [here](http://matplotlib.org/faq/virtualenv_faq.html) for more details. We recommend the Anaconda python distribution.
	
	- On MacOS, if you met the 'Value Error: unknown locale: UTF-8' error, fix by:
	
	        $ export LC_ALL=en_US.UTF-8
	        $ export LANG=en_US.UTF-8
	

**Note**: Both 32-bit and 64-bit programs are ok to build and run. But if users want to use the python wrapper, it's required that the architectures of ``python`` and LIBSOL are the same, i.e., 64-bit ``python`` can only use 64-bit LIBSOL, 32-bit ``python`` can only use 32-bit LIBSOL.

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



Comparison of Online Learning Algorithms
========================================
The ``experiments`` folder contains some scripts for a comparison of online
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
    pa1         0.8011+/-0.0058 0.0029+/-0.0029 0.8193+/-0.0103 0.0152+/-0.0013
    pa2         0.7913+/-0.0062 0.0018+/-0.0001 0.8013+/-0.0200 0.0146+/-0.0011
    eccw        0.7950+/-0.0067 0.0018+/-0.0001 0.7985+/-0.0097 0.0155+/-0.0016
    arow        0.8211+/-0.0061 0.0018+/-0.0001 0.8402+/-0.0009 0.0147+/-0.0009
    pa          0.7759+/-0.0097 0.0018+/-0.0001 0.7758+/-0.0329 0.0151+/-0.0011
    sop         0.7816+/-0.0073 0.0019+/-0.0001 0.7840+/-0.0189 0.0152+/-0.0007
    ada-fobos   0.8055+/-0.0052 0.0019+/-0.0001 0.8234+/-0.0043 0.0149+/-0.0009
    ada-rda     0.8114+/-0.0032 0.0019+/-0.0001 0.8347+/-0.0049 0.0147+/-0.0008
    rda         0.7528+/-0.0008 0.0019+/-0.0002 0.7595+/-0.0000 0.0145+/-0.0009
    erda        0.8049+/-0.0055 0.0019+/-0.0001 0.8326+/-0.0067 0.0146+/-0.0013
    cw          0.7913+/-0.0065 0.0018+/-0.0001 0.7907+/-0.0113 0.0149+/-0.0010
    vw          0.8443+/-0.0082 0.0571+/-0.0683 0.8326+/-0.0069 0.2582+/-0.0310
    alma2       0.8087+/-0.0037 0.0017+/-0.0001 0.8263+/-0.0089 0.0153+/-0.0013
    ogd         0.8108+/-0.0041 0.0019+/-0.0001 0.8363+/-0.0020 0.0150+/-0.0010
    perceptron  0.7713+/-0.0054 0.0017+/-0.0001 0.7793+/-0.0187 0.0151+/-0.0009
    liblinear   0.8536+/-0.0000 0.0356+/-0.0028 0.8425+/-0.0000 0.5620+/-0.0036

There will also be three pdf figures displaying the update number, training error rate, and test error rate over model sparsity.

Users can also compare on the multi-class dataset
[``mnist``](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist) with the follow command (Note that we only shuffle the training data once in this example, so the standard deviation is zero):

    $ python experiment.py mnist ../data/mnist.scale ../data/mnist.scale.t

The output is:

    algorithm   train           train           test            test
                accuracy        time(s)         accuracy        time(s)
    pa1         0.9857+/-0.0003 1.4516+/-0.0254 0.9903+/-0.0040 0.2372+/-0.0025
    pa2         0.9855+/-0.0003 1.4216+/-0.0182 0.9901+/-0.0038 0.2379+/-0.0038
    eccw        0.8640+/-0.0817 1.4180+/-0.0194 0.8673+/-0.0856 0.2386+/-0.0052
    arow        0.9906+/-0.0002 1.4110+/-0.0205 0.9945+/-0.0003 0.2366+/-0.0048
    perceptron  0.9832+/-0.0003 1.4166+/-0.0207 0.9890+/-0.0042 0.2383+/-0.0051
    ada-rda     0.9904+/-0.0002 1.4178+/-0.0207 0.9943+/-0.0003 0.2392+/-0.0054
    ada-fobos   0.9906+/-0.0001 1.4213+/-0.0167 0.9944+/-0.0004 0.2380+/-0.0042
    sop         0.9857+/-0.0003 1.4205+/-0.0212 0.9905+/-0.0042 0.2392+/-0.0021
    pa          0.9853+/-0.0003 1.4077+/-0.0292 0.9900+/-0.0041 0.2384+/-0.0026
    rda         0.9525+/-0.0007 1.4109+/-0.0108 0.9514+/-0.0015 0.2408+/-0.0062
    cw          0.9878+/-0.0002 1.4162+/-0.0170 0.9906+/-0.0031 0.2367+/-0.0033
    ogd         0.9884+/-0.0004 1.4165+/-0.0222 0.9930+/-0.0008 0.2359+/-0.0036
    alma2       0.9890+/-0.0002 1.4134+/-0.0101 0.9933+/-0.0005 0.2353+/-0.0041
    erda        0.9888+/-0.0002 1.4137+/-0.0214 0.9934+/-0.0009 0.2383+/-0.0055
    liblinear   0.9263+/-0.0001 145.5023+/-17.7614 0.9183+/-0.0001 2.0172+/-0.0164

The tables and figures in our paper description are obtained with the following
command:

    $ python experiment.py --shuffle 10 rcv1 ../data/rcv1_train ../data/rcv1_test

Additional Information
======================

For any questions and comments, please send your email to
chhoi@smu.edu.sg

Released date: 18 July, 2016.
