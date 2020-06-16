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
        self.d = 4
        self.n_del = 0
        self.n_del_list = 0
        self.delta = 0.01
        self.bipartite = False
        self.std = 0
        self.maxIter = 100
        self.n_del_init = 0
        self.path = 'results/odor_embdding/d4/' #'../results/tree/', '../results/missing_measurements/', ../results/sensitivity/ 'odor_embdding'
        self.solver = ''
        self.space = 'Euclidean' # 'Hyperbolic', 'Euclidean'
        self.experiment = 'missing_measurements' # 'tree', 'missing_measurements' 'sensitivity'
        self.load = False
        self.cost = 'TRACE' # 'TRACE', 'LOG-DET'
        self.norm = 'fro' # 'l1', 'l2', 'p1', 'fro'
        self.solver = 'CVXOPT' # 'CVXOPT', 'SCS'
        self.error_list = 10**(np.linspace(-2, 0, num=5))
        self.delta_list = 10**(np.linspace(0, -3, num=5))
param = parameters()
###########################################################################
#if param.experiment == 'tree':
#    range_of_N = range(10,11) # Number of nodes in a sequence of trees
#    M = 100 # Number of random trees
#    for N in range_of_N:
#        param.N = N
##        tree.tree_embedding(param,M)
#        print('N=', N)
###########################################################################
param.n_del_init = 1
if param.experiment == 'missing_measurements':
    range_of_N = range(9,27)
    for N in range_of_N:	
        param.N = N
        for n_del in range(param.n_del_init, htools.edgeCnt(param)):
            param.n_del = n_del
            prob = hdm.FindMaxSprsSGD(param)
            print('N=', N, 'n_del=', param.n_del,',d=', param.d,',p=',prob)
            if prob >= 0.9:
            	param.n_del_init = n_del
            if prob <= 0.1:
            	break
###########################################################################
if param.experiment == 'sensitivity':
    range_of_N = range(6,7)
    K = 4
    M1 = 100
    M2 = 10
    for N in range_of_N:
        print('N is ', N)
        param.N = N
        order.sensitivity(param,M1,M2,K)
###########################################################################
if param.experiment == 'odor_embdding':
    K = 4
    order.odor_embedding(param,K)
###########################################################################