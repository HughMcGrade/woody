# import sys
# sys.path.append(".")

import os
#import json
#from util import evaluate
import params

import time
import numpy as np
from woody.models import WoodClassifier

#from woody.io import  MemoryStore, DiskStore

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
                    n_queries,
                    n_features,
                    "empty(i32)",
                    0,
                    0,
                    nodes[len(nodes)-1][1]))

        
def call_futhark(programname, output_preds_fname, dkey, train_size):
    """Calls the futhark program with the given programname, with the generated tree as input"""
    cmd = 'cat tmp_tree | ./{} -t measurements/{} -r 10 > measurements/{}.txt'.format(programname,
                                                                                      "{}_{}_{}_times.txt".format(dkey,
                                                                                                                  train_size,
                                                                                                                  programname),
                                                                                      "{}_{}_{}".format(dkey,
                                                                                                        train_size,
                                                                                                        output_preds_fname))
    print("Command: {}".format(cmd))
    os.system(cmd)

def compare_predictions(cpu_pred, futhark_preds):
    """Compare predicitons between futhark and woody"""
    print ("Length of woody preds: {}".format(len(cpu_pred)))
    print ("Length of futhark preds: {}".format(len(futhark_preds)))

    correct = 0
    error = 0
    print ("First 10 entries of both:")
    print ("Futhark: {}".format(futhark_preds[:10]))
    print("Woody: {}".format(cpu_pred[:10]))
    
    for i in range(min(len(cpu_pred), len(futhark_preds))):
        if (cpu_pred[i] == futhark_preds[i]):
            correct += 1
        else:
            error += 1
            diff_sum += abs(cpu_pred[i]-futhark_preds[i])
    
    print ("Correct: {}\tErrors: {}".format(correct, error))

def get_futhark_predictions(dkey, train_size, fname):
    """Get recently predicted futhark predictions from disk"""
    futhark_preds = []
    
    with open(os.path.join("measurements", "{}_{}_{}".format(dkey, train_size, fname))) as file:
        for line in file.readlines():
            stripped = line.replace("[","").replace("i32","").replace(" ","").replace("]","").strip().split(",")
            futhark_preds.extend([float(i) for i in stripped])
    return futhark_preds

def cpu_prediction(n, Xtest, model):
    """
    Summary: Calls predict_single_tree n times. 
    Input:   
            n:     Number of times to call predict
            Xtest: Testdata to predict on
            model: WoodClassifier object that has a fitted forest
    Returns: list of execution times in microseconds"""
    times = []
    for i in range(n):
        cpu_pred_start_time = time.time()
        cpu_pred = super(WoodClassifier, model).predict_single_tree(Xtest)
        cpu_pred_stop_time = time.time()
        micro = int((cpu_pred_stop_time - cpu_pred_start_time) * 1000000)
        times.append(micro)
    return times
        
def single_run(dkey, train_size, param, seed, test_size=None):

    print("Processing data set %s with train_size %s, seed %s, and parameters %s ..." % (str(dkey), str(train_size), str(seed), str(param)))

    if dkey == "covtype":
        Xtrain, ytrain, Xtest, ytest = covtype(train_size=train_size, seed=seed)
    else:
        raise Exception("Unknown data set!")

    if test_size != None:
        for i in range(3):
            Xtest = np.concatenate((Xtest, Xtest), axis=0)
            ytest = np.concatenate((ytest, ytest), axis=0)
        Xtest = Xtest[:test_size]
        ytest = ytest[:test_size]
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


    ypreds_train = model.predict(Xtrain)
    # testing
    ypred_test = model.predict(Xtest)

    print("Calling predict_single_tree...")
    cpu_times = cpu_prediction(10, Xtest, model)
    
    cpu_pred = super(WoodClassifier, model).predict_single_tree(Xtest)


    
    print("After calling predict_single_tree...\nTimes in microseconds:")
    for t in cpu_times:
        print(t)

    # Save the cpu runtimes for use in report later
    with open("measurements/{}_{}_cpu_times.txt".format(dkey, train_size), "w") as file:
        for t in cpu_times:
            file.write("{}\n".format(t))
    # Save the predictions for comparisons
    print("Calling predict_single_tree again to save the predictions to file.")
    super(WoodClassifier, model).predict_single_tree_save_predictions(Xtest)
    print("Saving the first tree of the forest, so we can reload it in futhark")
    super(WoodClassifier, model).save_first_tree()

    time.sleep(1)


    print("Trying to generate futhark tree...")
    # This will read the tree from file,
    # restructure it for futhark and save as a new file called tmp_tree.  
    newfun(Xtest)

    # Call futhark!
    call_futhark("treesolver_basic", "basicout", dkey, train_size)
    call_futhark("treesolver", "pruneout", dkey, train_size)
    call_futhark("treesolver_flat", "flatout", dkey, train_size)
    call_futhark("treesolver_superflat", "superflatout", dkey, train_size)
    call_futhark("treesolver_precompute", "pre_out", dkey, train_size)


# ================================================================================
# evaluation of correctness
# ================================================================================


    print("Comparing basicout")
    futpreds = get_futhark_predictions(dkey, train_size, "basicout.txt")
    compare_predictions(cpu_pred, futpreds)

    print("Comparing flatout")
    futpreds = get_futhark_predictions(dkey, train_size, "flatout.txt")
    compare_predictions(cpu_pred, futpreds)

    print("Comparing pruneout")
    futpreds = get_futhark_predictions(dkey, train_size, "pruneout.txt")
    compare_predictions(cpu_pred, futpreds)

    print("Comparing superflatout")
    futpreds = get_futhark_predictions(dkey, train_size, "superflatout.txt")
    compare_predictions(cpu_pred, futpreds)

    print("Comparing pre_out")
    futpreds = get_futhark_predictions(dkey, train_size, "pre_out.txt")
    compare_predictions(cpu_pred, futpreds)

    # Delete the model after use
    del(model)
    
            
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

single_run(dkey, train_size, params.parameters[key], seed, train_size)
