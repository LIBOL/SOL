Experiments for comparison of different online learning algorithm
================================================================

The python scripts in this folder is for comparison of different online
learning algorithms, as well as the comparison with VW and LIBLINEAR. 


For example, to compare on the a1a dataset, you can simpy run:

    python experiment.py a1a ../data/a1a ../data/a1a.t


The algorithms to be compared are defined in "a1a.py". By default, we include
the comparison with VW and LIBLINEAR. You should have both packages properly
installed in your system. If **NOT**,  you can simply remove the configuration
for VW and LIBLINEAR from the configuration scripts.
