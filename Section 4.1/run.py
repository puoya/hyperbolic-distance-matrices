#!/usr/bin/env python
# coding=utf-8
import numpy as np
import os
import site
site.addsitedir('../lib/') 
import htools
import hdm
###########################################################################
class parameters:
    def __init__(self):
        self.N = 10
        self.d = 4
        self.n_del = 0
        self.delta = 0.01
        self.maxIter = 100
        self.n_del_init = 0
        self.path = '../results/missing_measurements/' #'../results/sensitivity/'
        self.solver = ''
        self.experiment = 'missing_measurements' # 'sensitivity'
        self.cost = 'TRACE' # 'TRACE', 'LOG-DET'
        self.norm = 'fro' # 'l1', 'l2', 'p1', 'fro'
        self.solver = 'CVXOPT' # 'CVXOPT', 'SCS'
        self.error_list = 10**(np.linspace(-2, 0, num=5))
        self.delta_list = 10**(np.linspace(0, -3, num=5))
param = parameters()
###########################################################################
param.n_del_init = 1
if param.experiment == 'missing_measurements':
    range_of_N = range(10,20)
    for N in range_of_N:
        param.N = N
        for n_del in range(param.n_del_init, htools.edgeCnt(param)):
            param.n_del = n_del
            prob = hdm.FindMaxSprs(param)
            print('N=', N, 'n_del=', param.n_del,',d=', param.d,',p=',prob)
###########################################################################
if param.experiment == 'sensitivity':
    range_of_N = range(5,20)
    R = 4
    K = 10
    M = 50
    for N in range_of_N:
        print('N is ', N)
        param.N = N
        hdm.sensitivity(param,K,M,R)