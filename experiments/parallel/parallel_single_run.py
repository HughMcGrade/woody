# import sys
# sys.path.append(".")

import os
import json
from util import evaluate
import params

import time

from woody.models import HugeWoodClassifier, WoodClassifier

from woody.io import  MemoryStore, DiskStore
from woody.util import ensure_dir_for_file
from woody.data import *

def single_run(dkey, train_size, param, seed):

    print("Processing data set %s with train_size %s, seed %s, and parameters %s ..." % (str(dkey), str(train_size), str(seed), str(param)))

    traingen, testgen = covtype_generators(train_size=train_size, store="mem", seed=seed)

    print("")
    print("Number of training patterns:\t%i" % traingen.get_shapes()[0][0])
    print("Number of test patterns:\t%i" % testgen.get_shapes()[0][0])
    print("Dimensionality of the data:\t%i\n" % traingen.get_shapes()[0][1])

    param_wood = param['param_wood']

    wood = WoodClassifier(
                n_estimators=1,
                criterion="gini",
                max_features=param_wood['max_features'],
                min_samples_split=2,
                n_jobs=param_wood['n_jobs'],
                seed=seed,
                bootstrap=param_wood['bootstrap'],
                tree_traversal_mode="dfs",
                tree_type=param_wood['tree_type'],
                min_samples_leaf=1,
                float_type="double",
                max_depth=None,
                verbose=0)
    top_tree_lambda = 0.1
    model = HugeWoodClassifier(
               n_estimators=param['n_estimators'],
               n_estimators_bottom=param['n_estimators_bottom'],
               n_top="auto",
               n_patterns_leaf="auto",
               balanced_top_tree=True,
               top_tree_lambda=top_tree_lambda,
               top_tree_max_depth=None,
               top_tree_type="standard",
               top_tree_leaf_stopping_mode="ignore_impurity",
               n_jobs=param_wood['n_jobs'],
               seed=seed,
               verbose=1,
               plot_intermediate={},
               chunk_max_megabytes=2048,
               wrapped_instance=wood,
               store=MemoryStore(),
               )

    # training

    fit_start_time = time.time()
    model.fit(traingen)
    fit_end_time = time.time()
    
# ==================================================
# This would be a brilliant place to save the first tree
# ==================================================

    # ypreds_train = model.predict(generator=traingen)

    # # testing
    # test_start_time = time.time()
    # ypred_test = model.predict(generator=testgen)
    # test_end_time = time.time()

    # results = {}
    # results['dataset'] = dkey
    # results['param'] = param
    # results['training_time'] = fit_end_time - fit_start_time
    # results['testing_time'] = test_end_time - test_start_time
    # print("Training time:\t\t%f" % results['training_time'])
    # print("Testing time:\t\t%f" % results['testing_time'])

    # evaluate(ypreds_train, traingen.get_all_target(), results, "training")
    # evaluate(ypred_test, testgen.get_all_target(), results, "testing")

    # fname = '%s_%s_%s_%s_%s_%s.json' % (str(param_wood['n_estimators']),
    #                               str(param_wood['max_features']),
    #                               str(param_wood['n_jobs']),
    #                               str(param_wood['bootstrap']),
    #                               str(param_wood['tree_type']),
    #                               str(seed),
    #                             )
    # fname = os.path.join(params.odir, str(dkey), str(train_size), "hugewood_" + str(top_tree_lambda), fname)
    # ensure_dir_for_file(fname)
    # with open(fname, 'w') as fp:
    #     json.dump(results, fp)

    del(testgen)
    del(traingen)
    model.cleanup()

    time.sleep(1)

###################################################################################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dkey', nargs='?', const="covtype", type=str, default="covtype")
parser.add_argument('--train_size', nargs='?', const=0, type=int, default=0)
parser.add_argument('--seed', nargs='?', const=0, type=int, default=0)
parser.add_argument('--key', type=str)
args = parser.parse_args()
dkey, train_size, seed, key = args.dkey, args.train_size, args.seed, args.key
###################################################################################

single_run(dkey, train_size, params.parameters_hugewood[key], seed)
