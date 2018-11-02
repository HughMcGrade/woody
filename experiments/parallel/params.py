import collections

odir = "results"
methods = ["hugewood_lam"]

datasets = collections.OrderedDict()
datasets['covtype'] = {'train_sizes':[1000, 10000, 100000, 200000, 300000, 400000]}
datasets["susy"] = {'train_sizes':[1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000]}
datasets["higgs"] = {'train_sizes':[1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000]}

parameters = collections.OrderedDict()
parameters['rf'] = {'n_estimators':24,
                    'max_features':"sqrt", 
                    'bootstrap':True, 
                    'tree_type':'standard', 
                    'n_jobs':1}

parameters_hugewood = collections.OrderedDict()

for key in parameters:
    
    param_hugewood = {}
    param_hugewood['param_wood'] = parameters[key]
    param_hugewood['n_estimators'] = 6
    param_hugewood['n_estimators_bottom'] = 4
    
    parameters_hugewood[key] = param_hugewood
