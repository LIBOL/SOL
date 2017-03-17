Experimental Scripts for Large-scale Online Feature Selection for Ultra-high Dimensional Sparse Data
================================================================

The python scripts in this folder are for the following paper:

    Yue Wu, Steven C.H. Hoi, Tao Mei, and Nenghai Yu. 2017. Large-scale Online Feature Selection
    for Ultra-high Dimensional Sparse Data. ACM Transactions on Knowledge Discovery from Data.


# Installation

1. Install the SOL python scripts. Refer to the SOL documentation for details.

2. Some external packages you may need to install for full set of experiments:

    + [fast-mRMR](https://github.com/sramirez/fast-mRMR)

        1. Comile the cpu version for general mRMR algorithm.

        2. If you have a Nvidia GPU card, compile the gpu version.

        3. Compile the data-reader in utils. **Change the arguments of the main
        function as follows, so that the output file is specified manually.**

                line 77: ofstream outputFile(argv[2], ios::out | ios::binary);

        4. Add the **fast-mrmr**, **gpu-mrmr**, and **mrmr-reader** to the
           system path.

    + [FGM](http://www.tanmingkui.com/fgm.html)

        Compile the code and add the executable **FGM** to the system path.

# Experiments

The configuarions for the datasets are in the **"opts"** folder.
For example, to compare performance on the 'aut' dataset, you can simpy run:

    python fs.py aut /data/sol/aut/aut_train /data/sol/aut/aut_test

The results and figures will be saved to the folder **"cache/aut"**.
