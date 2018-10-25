# self.wrapper.module.predict_extern(X, preds, indices, self.wrapper.params, self.wrapper.forest)
# preds_fut = treesolve(X, preds_fut, indices, self.wrapper.params, self.wrapper.forest, preds)
from treesolver import treesolver
import numpy as np

solver = treesolver()

def treesolve(X, indices, tree):
    print("X = " + str(X))
    print("indices = " + str(indices))
    print("tree = " + str(tree[0].left_id))
    tree_size = len(tree)
    left_ids = np.zeros(tree_size, dtype='int')
    right_ids = np.zeros(tree_size, dtype='int')
    features = np.zeros(tree_size, dtype='int')
    thres_or_leaf = np.zeros(tree_size)
    for i in range(tree_size):
        left_ids[i] = tree[i].left_id
        right_ids[i] = tree[i].right_id
        features[i] = tree[i].feature
        thres_or_leaf[i] = 0.5
    Xtest = np.random.rand(10, 10)
    nXtest = 10
    dXtest = 10
    indices = np.zeros(0, dtype='int')
    dindices = 0
    print("futhark says: " + str(solver.main(left_ids, right_ids, features, thres_or_leaf, Xtest, nXtest, dXtest, indices, dindices, 0)))
    return 0

def forest_to_arrays(forest):
    pass
