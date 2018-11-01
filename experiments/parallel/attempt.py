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


def build_wood(param_wood):
    return WoodClassifier(
                n_estimators=1,
                criterion="gini",
#                max_features=param_wood['max_features'],
                max_features="sqrt",        
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

def build_model(param_wood, wrapped_instance, param, top_tree_lambda):
    return HugeWoodClassifier(
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
               wrapped_instance=wrapped_instance,
               store=MemoryStore(),
               )

def single_run(dkey, train_size, param, seed):

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
    # print("Number of training patterns:\t%i" % traingen.get_shapes()[0][0])
    # print("Number of test patterns:\t%i" % testgen.get_shapes()[0][0])
    # print("Dimensionality of the data:\t%i\n" % traingen.get_shapes()[0][1])
    print("Number of training patterns:\t%i" % Xtrain.shape[0])
    print("Number of test patterns:\t%i" % Xtest.shape[0])
    print("Dimensionality of the data:\t%i\n" % Xtrain.shape[1])

    # param_wood = param['param_wood']

    # wood = build_wood(param_wood)
    # top_tree_lambda = 0.1
    # model = build_model(param_wood, wood, param, top_tree_lambda)
    model = WoodClassifier(
                n_estimators=param['n_estimators'],
                criterion="gini",
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
                verbose=0)

    # training
    model.fit(Xtrain, ytrain)

# ==================================================
# This would be a brilliant place to save the first tree
# ==================================================

    ypreds_train = model.predict(Xtrain)
    # testing

    ypred_test = model.predict(Xtest)
#    wrap = super(WoodClassifier, wood).get_wrapper()

    print("Calling predict_single_tree...")
    cpu_pred_start_time = time.time()
    tmpPred = super(WoodClassifier, model).predict_single_tree(Xtest)
    cpu_pred_stop_time = time.time()
    print("After calling predict_single_tree...")
    print("CPU Call took: %f" % (cpu_pred_stop_time - cpu_pred_start_time))
    print("tmpPred:")
    print(tmpPred)
    print("Calling predict_single_tree again to save the predictions to file.")
    super(WoodClassifier, model).predict_single_tree_save_predictions(Xtest)
    super(WoodClassifier, model).save_first_tree()
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

single_run(dkey, train_size, params.parameters[key], seed)#_hugewood[key], seed)
