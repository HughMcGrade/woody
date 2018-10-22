import os
import params

seeds = [0,1,2,3]
odir = params.odir
methods = params.methods

for method in methods[:1]:
    for dkey in params.datasets.keys()[:1]:
        for train_size in params.datasets[dkey]['train_sizes'][:1]:
            for seed in seeds[:1]:
                for key in params.parameters:
                    print("Processing method %s with data set %s, train_size %s, seed %s, and key %s ..." % (str(method), str(dkey), str(train_size), str(seed), str(key)))
                    cmd = "python " + method + ".py --dkey %s --train_size %i --seed %i --key %s" % (dkey, train_size, seed, key)
                    print(cmd)
                    os.system(cmd)
