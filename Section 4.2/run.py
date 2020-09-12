#!/usr/bin/env python
# coding=utf-8
import numpy as np
import tree
###########################################################################
class parameters:
    def __init__(self):
        self.N = 10
        self.d = 2
        self.maxIter = 100
        self.path = '../results/tree/'
        self.space = 'Hyperbolic' # 'Hyperbolic', 'Euclidean'
        self.cost = 'LOG-DET' 
        self.norm = 'fro' # 'l1', 'l2', 'p1', 'fro'
        self.solver = 'CVXOPT' # 'CVXOPT', 'SCS'
        self.error_list = 10**(np.linspace(-2, 0, num=5))
        self.delta_list = 10**(np.linspace(0, -3, num=5))
param = parameters()
###########################################################################
range_of_N = range(10,11) # Number of nodes in a sequence of trees
M = 100 # Number of random trees
for N in range_of_N:
    param.N = N
    tree.tree_embedding(param,M)
    print('N=', N)