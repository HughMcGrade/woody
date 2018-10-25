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
    left_ids = np.zeros((tree_size))
    right_ids = np.zeros((tree_size))
    features = np.zeros((tree_size))
    thres_or_leaf = np.zeros((tree_size))
    for i in range(tree_size):
        left_ids[i] = tree[i].left_id
        right_ids[i] = tree[i].right_id
        features[i] = tree[i].feature
        #thres_or_leaf[i] = tree[i].thres_or_leaf
        #print(dir(tree[i].thres_or_leaf))
        #print(tree[i].thres_or_leaf)
        #acquired = tree[i].thres_or_leaf.acquire()
        #print(acquired)
        print(tree[i].thres_or_leaf.value())
        #print(dir(tree[i].thres_or_leaf))
        #print((left_ids[i], right_ids[i], features[i], thres_or_leaf[i]))
    print("futhark says: " + str(solver.main(1, 2, 3)))
    return 0

def forest_to_arrays(forest):
    pass
