#!/usr/bin/env python
# coding=utf-8
import hdm
import htools
import numpy as np
import matplotlib.pyplot as plt
import tree
import order
import os
trj_output = []
kedm_output = []
###########################################################################
class parameters:
    def __init__(self):
        self.N = 10
        self.d = 2
        self.n_del = 1125
        self.n_del_list = 0
        self.delta = 0.9
        self.bipartite = False
        self.std = 0
        self.maxIter = 100
        self.n_del_init = 0
        self.path = '../results/odor_embdding/' #'../results/tree/', '../results/missing_measurements/', ../results/sensitivity/ 'odor_embdding'
        self.solver = ''
        self.space = 'Euclidean' # 'Hyperbolic', 'Euclidean'
        self.experiment = 'odor_embdding' # 'tree', 'missing_measurements' 'sensitivity'
        self.load = False
        self.cost = 'TRACE' # 'TRACE', 'LOG-DET'
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