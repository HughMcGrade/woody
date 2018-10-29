import collections

odir = "results"
methods = ["hugewood_lam"]

datasets = collections.OrderedDict()
datasets['covtype'] = {'train_sizes':[100000]}

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
