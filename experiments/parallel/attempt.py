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

def newfun(Xtest):
    n_queries, n_features = len(Xtest), len(Xtest[0])
    data_headers = [("features", int), ("leaf_criterion", int), ("left_ids", int), ("right_ids",int), ("thres_or_leaf",float)]
    data_values = {}
    query_values = []
    
    amount_of_queries = n_queries
    amount_of_features = n_features

    for header in data_headers:
        print header
        with open(os.path.join(header[0]+".txt")) as file:
            data_values[header[0]] = [header[1](i) if header[1](i)<10000000 else 0 for i in file.readline().replace("[","").replace("]","").replace(",","").strip().split(" ")[:-1]]

    data_values["right_ids"]

    #Crawl the tree
    current_depth = 0
    max_depth = 0
    nodes = [(0,0)]

    for node in nodes:
        if ((data_values["left_ids"][node[0]] != 0) and (data_values["right_ids"][node[0]] != 0)):
            nodes.append((data_values["left_ids"][node[0]],node[1]+1))
            nodes.append((data_values["right_ids"][node[0]],node[1]+1))


    for row in Xtest:
        query_values.extend([int(i) for i in row])
            
    with open("tmp_tree",'w') as file:
        file.write('{} {} {} {} {} {} {} {} {} {} {}'.format(\
                    str(data_values["left_ids"]),
                    str(data_values["right_ids"]),
                    str(data_values["features"]),
                    str(data_values["thres_or_leaf"]),
                    str(query_values),
                    amount_of_queries,
                    amount_of_features,
                    "empty(i32)",
                    0,
                    0,
                    nodes[len(nodes)-1][1]))


def compare_predictions(cpu_pred, futhark_preds):
    print ("Length of woody preds: {}".format(len(cpu_pred)))
    print ("Length of futhark preds: {}".format(len(futhark_preds)))

    correct = 0
    error = 0
    diff_sum = 0
    print ("First 10 entries of both:")
    print ("Futhark: {}".format(futhark_preds[:10]))
    print("Woody: {}".format(cpu_pred[:10]))
    
    for i in range(min(len(cpu_pred), len(futhark_preds))):
        if (cpu_pred[i] == futhark_preds[i]):
            correct += 1
        else:
            error += 1
            diff_sum += abs(cpu_pred[i]-futhark_preds[i])
    avg = 0 if (error == 0) else diff_sum/error
    print ("Correct: {}\tErrors: {}\tAverage error: {}".format(correct, error, avg))

def get_futhark_predictions(fname):
    futhark_preds = []
    with open(fname) as file:
        for line in file.readlines():
            stripped = line.replace("[","").replace("i32","").replace(" ","").replace("]","").strip().split(",")
            futhark_preds.extend([float(i) for i in stripped])
    return futhark_preds

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
    print("Number of training patterns:\t%i" % Xtrain.shape[0])
    print("Number of test patterns:\t%i" % Xtest.shape[0])
    print("Dimensionality of the data:\t%i\n" % Xtrain.shape[1])

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
    cpu_pred = super(WoodClassifier, model).predict_single_tree(Xtest)
    cpu_pred_stop_time = time.time()
    print("After calling predict_single_tree...")
    print("CPU Call took: %f" % (cpu_pred_stop_time - cpu_pred_start_time))
    print("Type of cpu_pred: {}".format(type(cpu_pred)))
    
    print("Calling predict_single_tree again to save the predictions to file.")
    super(WoodClassifier, model).predict_single_tree_save_predictions(Xtest)
    print("Saving the first tree of the forest, so we can reload it in futhark")
    super(WoodClassifier, model).save_first_tree()
    time.sleep(1)


    print("Trying to generate futhark tree...")
    # This will read the tree from file,
    # restructure it for futhark and save as a new file called tmp_tree.  
    newfun(Xtest)
    
    print ("Trying to call futhark from python")
    futstart = time.time()
    cmd = 'cat tmp_tree | ./treesolver_basic > basicout.txt'
    print("Command: {}".format(cmd))
    os.system(cmd)
    futstop = time.time()
    print("Futhark call took: %f" % (futstop - futstart))

    print ("Trying to call futhark from python")
    futstart = time.time()
    cmd = 'cat tmp_tree | ./treesolver > pruneout.txt'
    print("Command: {}".format(cmd))
    os.system(cmd)
    futstop = time.time()
    print("Futhark call took: %f" % (futstop - futstart))

    print ("Trying to call futhark from python")
    futstart = time.time()
    cmd = 'cat tmp_tree | ./treesolver_flat > flatout.txt'
    print("Command: {}".format(cmd))
    os.system(cmd)
    futstop = time.time()
    print("Futhark call took: %f" % (futstop - futstart))

    print ("Trying to call futhark from python")
    futstart = time.time()
    cmd = 'cat tmp_tree | ./treesolver_superflat > superflatout.txt'
    print("Command: {}".format(cmd))
    os.system(cmd)
    futstop = time.time()
    print("Futhark call took: %f" % (futstop - futstart))

    print ("Trying to call futhark from python")
    futstart = time.time()
    cmd = 'cat tmp_tree | ./treesolver_precompute > pre_out.txt'
    print("Command: {}".format(cmd))
    os.system(cmd)
    futstop = time.time()
    print("Futhark call took: %f" % (futstop - futstart))


    # sum = 0
    # for i in range(10):
    #     futstart = time.time()
    #     cmd = 'cat tmp_tree | ./treesolver_flat > flatout.txt'
    #     print("Command: {}".format(cmd))
    #     os.system(cmd)
    #     futstop = time.time()
    #     sum += (futstop - futstart)
    # avg = sum / 10.0
    # print("Average of 10 calls: {}".format(avg))

# ================================================================================
# evaluation of correctness
# ================================================================================


    print("Comparing basicout")
    futpreds = get_futhark_predictions("basicout.txt")
    compare_predictions(cpu_pred, futpreds)

    print("Comparing flatout")
    futpreds = get_futhark_predictions("flatout.txt")
    compare_predictions(cpu_pred, futpreds)

    print("Comparing pruneout")
    futpreds = get_futhark_predictions("pruneout.txt")
    compare_predictions(cpu_pred, futpreds)

    print("Comparing superflatout")
    futpreds = get_futhark_predictions("superflatout.txt")
    compare_predictions(cpu_pred, futpreds)

    print("Comparing pre_out")
    futpreds = get_futhark_predictions("pre_out.txt")
    compare_predictions(cpu_pred, futpreds)



    # print ("Length of woody preds: {}".format(len(cpu_pred)))
    # print ("Length of futhark preds: {}".format(len(futhark_preds)))

    # correct = 0
    # error = 0
    # diff_sum = 0
    # print ("First 10 entries of both:")
    # print ("Futhark: {}".format(futhark_preds[:10]))
    # print("Woody: {}".format(cpu_pred[:10]))
    
    # for i in range(min(len(cpu_pred), len(futhark_preds))):
    #     if (cpu_pred[i] == futhark_preds[i]):
    #         correct += 1
    #     else:
    #         error += 1
    #         diff_sum += abs(cpu_pred[i]-futhark_preds[i])
    # print ("Correct: {}\tErrors: {}\tAverage error: {}".format(correct, error, (diff_sum / error)))
    
            
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
