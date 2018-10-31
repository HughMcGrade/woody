import os
import params

seeds = [0,1,2,3]
odir = params.odir
methods = params.methods


cmd = "python parallel_single_run.py --dkey covtype --train_size 100 --seed 0 --key rf"
print "Running command: %s" % cmd
os.system(cmd)
print "finished running..."

def query_single_tree_woody(tree):
    pass

def query_single_tree_futhark(tree):
    pass
