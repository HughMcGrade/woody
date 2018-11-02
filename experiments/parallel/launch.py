import os
import params

seeds = [0,1,2,3]
odir = params.odir
methods = params.methods

for train_size in params.datasets['covtype']['train_sizes']:
    cmd = 'python compare_and_measure.py --dkey covtype --train_size {} --seed 0 --key rf'.format(train_size)
    print("Running command: {}".format(cmd))
    os.system(cmd)
print "finished running..."
