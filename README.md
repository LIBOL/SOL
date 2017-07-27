
SOL - A Library for Scalable Online Learning Algorithms
============================================


About SOL
====================================================
SOL is an open-source library for scalable online learning with high-dimensional data. The library provides a family of regular and sparse online learning algorithms for large-scale binary and multi-class classification tasks with high efficiency, scalability, portability, and extensibility. We provide easy-to-use command-line tools, python wrappers and library calls for users and developers, and comprehensive documents for both beginners and advanced users. SOL is not only a machine learning toolbox, but also a comprehensive experimental platform for online learning research. Experiments demonstrate that SOL is highly efficient and scalable for large-scale learning with high-dimensional data.

Specifically, SOL consists of a family of:
+ First-order Online Learning algorithms:
    - Perceptron: The Perceptron Algorithm(Rosenblatt, 1958)
    - OGD: Online Gradient Descent(Zinkevich, 2003)
    - PA: Online Passive Aggressive Algorithms(Crammer et al., 2006)
    - ALMA: Approximate Large Margin Algorithm(Gentile, 2002)
    - RDA: Regularized Dual Averaging(Xiao, 2010)
+ Second-order Online Learning algorithms:
    - SOP: Second-Order Perceptron(Cesa-Bianchi et al., 2005)
    - CW: Confidence Weighted Learning(Dredze et al., 2008)
    - ECCW: Exactly Convex Confidence Weighted Learning(Crammer et al., 2008)
    - AROW: Adaptive Regularized Online Learning(Crammer et al., 2009)
    - Ada-FOBOS: Adaptive Regularized Online Learning(Crammer et al., 2009)
    - Ada-RDA: Adaptive Regularized Dual Averaging(Crammer et al., 2009)

+ First-order Sparse Online Learning algorithms:
    - STG: sparse online learning via truncated graidient (Langford et al., 2009);
    - FOBOS-L1: l1 Regularized Forward backward splitting (Duchi et al., 2009);
    - RDA-L1: Mixed l1/l2^2 Regularized Dual averaging(Xiao, 2010);
    - ERDA-L1: Enhanced l1/l2^2 Regularized Dual averaging(Xiao, 2010);

+ Second-order Sparse Online Learning algorithms:
    - Ada-FOBOS-L1: Ada-FOBOS with l1 regularization
    - Ada-RDA-L1: Ada-RDA with l1 regularization

This document briefly explains the usage of SOL. A more detailed manual for users and developers can be
found in the documentation of SOL.

To get started, please read the ``Quick Start'' section first.

Table of Contents
=================
- Installation
- Quick Start
- Additional Information

Installation
======================

Users can either install the C++ executables or python scripts. They provide similar interfaces (**sol_train** for training and **sol_test** for testing). To choose the best method:

+ If you are working on C/C++ or other non-python languages, you should choose the C++
  executables and dynamic libraries.

+ If you are a python worker, then just try the python scripts.

**NOTE**:

Both the python scripts and C++ executables & Libraries are dependent on the same C++ codes.

## Install from Source

SOL features a very simple installation procedure. The project is managed by `CMake` for C++ and `setuptools` for python.


###Getting the code

 There exists a `CMakeLists.txt` in the root directory.
The latest version of SOL is always available via 'github' by invoking one
of the following:

    ## For the traditional ssh-based Git interaction:
    $ git clone git://github.com/LIBOL/SOL.git

    ## For HTTP-based Git interaction
    $ git clone https://github.com/LIBOL/SOL.git

###Build C++ Executables and Dynamic Libraries

1. Prerequisites

    + CMake  2.8.12 or higher
    + Git
    + C++11 Compiler:
        - g++(>4.8.2) or clang++(>3.3) on Unix/Linux/MinGW/Cygwin
        - Visual Studio 2013 or higher on Windows

2. Make a folder to store project files:

        $ cd SOL && mkdir build && cd build

3. Generate and build the project files

    + For Unix/Linux/MacOS users

            $ cmake ..
            $ make -j
            $ make install


    + For users with Xcode

            $ cmake -G"Xcode" ..

    + For windows users

            $ cmake -G "Visual Studio 14 2015 Win64" ..

        Open the project ``SOL.sln``, Rebuild the `ALL_BUILD` project and then build the `INSTALL` project


4. The generated package will be copied to $SOL/dist

### Build Python Scripts

We highly recommend users to install python packages in a virtual enviroment.

1. Prerequisites

    + C++11 Compiler:
        - g++(>4.8.2) or clang++(>3.3) on Unix/Linux/MinGW/Cygwin
        - Visual Studio 2013 or higher on Windows
    + Python2.7 (Anaconda is highly recommend on Windows)

1. For on Unix/Linux/MacOS Users

    + Create and Activate a new virtual environment

            $ virtualenv --system-site-packages pyenv
            $ source pyenv/bin/activate

    + Build and install the python scripts

            $ python setup.py build
            $ python setup.py install

2. For Windows Users

    Windows Users still need **Visual Studio** to compile the codes. Besides,
    we highly recommend to install [Anaconda](https://www.continuum.io/) as the
    python environment. If you are a [Canopy](https://www.enthought.com/) user,
    the procedures are similar except for the creation of virtual environment.

    + Open a Command Prompt and go to the source directory

            $ cd <SOL>

    + Create and Activate a new virtual enviroment

            $ virtualenv --system-site-packages pyenv
            $ pyenv/Scripts/activate

    + Set up the build environment

        By default, Anaconda requires the Visual C++ 9.0 (Visual Studio 2008) to compile the native
        codes. However, this is a very old compiler. We recommend to use the
        following tricks  to use new visual studio compilers.

            $ cd <Anaconda>\Lib\distutils
            $ backup msvc9compiler.py
            $ open msvc9compiler.py
            $ find the line 'majorVersion=int(s[:2]) - 6'
            $ change the line to 'majorVersion=12' for Visual Studio 2013 or
            $ change the line to 'majorVersion=14' for Visual Studio 2015 or

    + Build and install the python scripts

            $ python setup.py build
            $ python setup.py install

    + Revert the changes to Anaconda if you are not sure its influences in the
      future.

## Known Issues of Python Wrappers

- On MacOS, the default python is not a framework build.  Seems matplotlib does not work properly. See [here](http://matplotlib.org/faq/virtualenv_faq.html) for more details. We recommend the Anaconda python distribution.

- On MacOS, if you met the 'Value Error: unknown locale: UTF-8' error, fix by:

        $ export LC_ALL=en_US.UTF-8
        $ export LANG=en_US.UTF-8



Quick Start
===========
Running SOL without any arguments or with '--help' will produce a message which briefly explains the arguments.

We provide an example to show how to use SOL and explain the details of how SOL works.
The dataset we use will be `a1a` provided in the ``data`` folder.

The command for training wit default algorithm is as the following shows.

    $ sol_train data/a1a
    training accuracy: 0.8125
    training time: 0.000 seconds
    model sparsity: 15.1260%

The learned model can be saved to a file (``a1a.model`` for example) by:

    $ sol_train data/a1a a1a.model

By default, SOL use ``OGD`` to learn a model. If users want to try another
algorithm (``AROW`` for example) and save to another file (``arow.model``):

    $ sol_train -a arow data/a1a arow.model

Each algorithm may have its own parameters. The following command changes the
default value of parameter ``r`` to ``2.0``:

    $ sol_train --params r=2.0 -a arow data/a1a arow.model

The python scripts also provides the cross validation ability. For example, if
users want to do a 5-fold GridSearch Cross Validation in the range [2^-5,2^-4,...,2^4, 2^5] for
parameter ``r`` of AROW, the command will be:

    $ sol_train -a arow --cv r=0.03125:2:32 -f 5 data/a1a arow.model
    cross validation parameters: [('r', 2.0)]

In some cases we want to finetune from a pretrained model,

    $ sol_train -m arow.model data/a1a arow2.model

We can test with the learned model:

    $ sol_test arow.model data/a1a.t predict.txt
    test accuracy: 0.8437
    test time: 0.016 seconds

**NOTES**

+  They python scripts will analyze the dataset (number of classes, dimension
   of the data) before training.

+ The C++ executable needs to be specified with number of classes for
  multi-class problems.

        $ sol_train -c 10 mnist.scale

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
    pa1         0.8553+/-0.0009 1.4840+/-0.0163 0.8753+/-0.0164 0.2474+/-0.0018
    pa2         0.8585+/-0.0008 1.4804+/-0.0228 0.8811+/-0.0131 0.2469+/-0.0040
    eccw        0.8764+/-0.0210 1.4825+/-0.0167 0.8688+/-0.0493 0.2468+/-0.0030
    arow        0.9051+/-0.0004 1.4680+/-0.0136 0.9226+/-0.0014 0.2458+/-0.0038
    perceptron  0.8460+/-0.0008 1.4752+/-0.0151 0.8671+/-0.0164 0.2483+/-0.0052
    ada-rda     0.8999+/-0.0005 1.4736+/-0.0090 0.9201+/-0.0022 0.2467+/-0.0047
    ada-fobos   0.9055+/-0.0007 1.4807+/-0.0196 0.9239+/-0.0020 0.2468+/-0.0027
    pa          0.8553+/-0.0009 1.4667+/-0.0210 0.8753+/-0.0164 0.2465+/-0.0023
    sop         0.8552+/-0.0007 1.4751+/-0.0149 0.8811+/-0.0099 0.2481+/-0.0054
    rda         0.7868+/-0.0009 1.4630+/-0.0166 0.8027+/-0.0027 0.2483+/-0.0018
    cw          0.8784+/-0.0008 1.4784+/-0.0240 0.8861+/-0.0034 0.2459+/-0.0044
    vw          0.9138+/-0.0023 1.8747+/-0.0743 0.9125+/-0.0019 0.3252+/-0.0121
    ogd         0.8943+/-0.0009 1.4724+/-0.0192 0.9171+/-0.0008 0.2481+/-0.0039
    alma2       0.8972+/-0.0008 1.4723+/-0.0134 0.9188+/-0.0022 0.2498+/-0.0118
    erda        0.8839+/-0.0005 1.4707+/-0.0120 0.9132+/-0.0042 0.2474+/-0.0017
    liblinear   0.9263+/-0.0001 145.5023+/-17.7614 0.9183+/-0.0001 2.0172+/-0.0164

The tables and figures in our paper description are obtained with the following
command:

    $ python experiment.py --shuffle 10 rcv1 ../data/rcv1_train ../data/rcv1_test


License and Citation
======================

SOL is released under the Apache 2.0 open source license.

SOL has been published as Original Software Publication in Neurocomputing Journal

Volume 260, 18 October 2017

"SOL: A library for scalable online learning algorithms"

Yue Wu, Steven C.H. Hoi, Chenghao Liu , Jing Lu, Doyen Sahoo, Nenghai Yu,

https://doi.org/10.1016/j.neucom.2017.03.077

Please cite SOL in your publications if it helps:

```
@article{sol2017,
  title={SOL: A Library for Scalable Online Learning Algorithms},
  author={Yue Wu, Steven C.H. Hoi, Chenghao Liu, Jing Lu, Doyen Sahoo, Nenghai Yu},
  journal={Neurocomputing},
  vol = {260},
  pages = {9--12},
  year={2017}
}
```

Additional Information
======================

For any questions and comments, please send your email to
chhoi@smu.edu.sg

Released date: 25 July, 2016.
