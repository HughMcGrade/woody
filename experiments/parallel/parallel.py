import os
import params

seeds = [0,1,2,3]
odir = params.odir
methods = params.methods


#cmd = "python parallel_single_run.py --dkey covtype --train_size 400000 --seed 0 --key rf"
# cmd = "python parallel_single_run.py --dkey covtype --train_size 350000 --seed 0 --key rf"
# cmd = "python parallel_single_run.py --dkey covtype --train_size 300000 --seed 0 --key rf"
# cmd = "python parallel_single_run.py --dkey covtype --train_size 250000 --seed 0 --key rf"
# cmd = "python parallel_single_run.py --dkey covtype --train_size 200000 --seed 0 --key rf"
# cmd = "python parallel_single_run.py --dkey covtype --train_size 150000 --seed 0 --key rf"
#cmd = "python attempt.py --dkey covtype --train_size 100000 --seed 0 --key rf"
cmd = "python attempt.py --dkey covtype --train_size 1000 --seed 0 --key rf"
print "Running command: %s" % cmd
os.system(cmd)
print "finished running..."

def query_single_tree_woody(tree):
    pass

def query_single_tree_futhark(tree):
    pass
