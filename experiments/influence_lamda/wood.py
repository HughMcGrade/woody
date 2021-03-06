import sys
sys.path.append(".")

import params
from util import evaluate

import os
import time
import json

from woody import WoodClassifier
from woody.util import ensure_dir_for_file
from woody.data import *
            
def single_run(dkey, train_size, lamcrit, param, seed, profile=False):     
           
    print("Processing data set %s with train_size %s, seed %s, and parameters %s ..." % (str(dkey), str(train_size), str(seed), str(param)))

     
    if dkey == "covtype":
        Xtrain, ytrain, Xtest, ytest = covtype(train_size=train_size, seed=seed)
    elif dkey == "higgs":
        Xtrain, ytrain, Xtest, ytest = higgs(train_size=train_size, seed=seed)
    elif dkey == "susy":
        Xtrain, ytrain, Xtest, ytest = susy(train_size=train_size, seed=seed)
    else:
        raise Exception("Unknown data set!")
    
    print("")
    print("Number of training patterns:\t%i" % Xtrain.shape[0])
    print("Number of test patterns:\t%i" % Xtest.shape[0])
    print("Dimensionality of the data:\t%i\n" % Xtrain.shape[1])
       
    model = WoodClassifier(
                n_estimators=param['n_estimators'],
                criterion="even_gini",
                max_features=param['max_features'],
                min_samples_split=2,
                n_jobs=param['n_jobs'],
                seed=seed,
                bootstrap=param['bootstrap'],
                tree_traversal_mode="dfs",
                tree_type=param['tree_type'],
                min_samples_leaf=1,
                float_type="double",
                max_depth=None,
                lam_criterion=lamcrit,
                verbose=0)
    
    if profile == True:
        import yep
        assert param['n_jobs'] == 1
        yep.start("train.prof")
                    
    # training
    fit_start_time = time.time()
    model.fit(Xtrain, ytrain)
    fit_end_time = time.time()
    if profile == True:
        yep.stop()            

    print("Number of nodes: %i" % model.get_n_nodes(0)) 
    ypreds_train = model.predict(Xtrain) 
     
    # testing
    test_start_time = time.time()
    ypred_test = model.predict(Xtest)
    test_end_time = time.time()
    
    results = {}
    results['dataset'] = dkey
    results['param'] = param
    results['training_time'] = fit_end_time - fit_start_time
    results['testing_time'] = test_end_time - test_start_time
    print("Training time:     %f" % results['training_time'])
    print("Testing time:      %f" % results['testing_time'])
                
    evaluate(ypreds_train, ytrain, results, "training")
    evaluate(ypred_test, ytest, results, "testing")
                    
    fname = '%s_%s_%s_%s_%s_%s.json' % (str(param['n_estimators']),
                                  str(param['max_features']),
                                  str(param['n_jobs']),
                                  str(param['bootstrap']),
                                  str(param['tree_type']),
                                  str(seed),
                                )
    fname = os.path.join(params.odir, str(dkey), str(train_size), str(lamcrit), "wood", fname)
    ensure_dir_for_file(fname)
    with open(fname, 'w') as fp:
        json.dump(results, fp)
            
###################################################################################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dkey', nargs='?', const="covtype", type=str, default="covtype")
parser.add_argument('--train_size', nargs='?', const=0, type=int, default=0)
parser.add_argument('--seed', nargs='?', const=0, type=int, default=0)
parser.add_argument('--key', type=str)
parser.add_argument('--lamcrit', nargs='?', const=0.0, type=float, default=0.0)
args = parser.parse_args()
dkey, train_size, seed, key, lamcrit = args.dkey, args.train_size, args.seed, args.key, args.lamcrit
###################################################################################

single_run(dkey, train_size, lamcrit, params.parameters[key], seed)            
            
