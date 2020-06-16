#!/usr/bin/env python
# coding=utf-8
import hdm
import htools
import numpy as np
import os
def sensitivity(param,M1,M2,K):
    directory = param.path
    N= param.N
    index = htools.ordinal_set(param)
    N_choose_2 = int(N*(N-1)/2)
    N_choose_2_2 = int( N_choose_2 * ( N_choose_2-1)/2 )
    S = 1-K*N_choose_2/N_choose_2_2
    print('Sparsity is ', S)
    Dt = np.zeros((M2,M1,N,N))
    for i in range(M2):    
        X = htools.randX(param)
        D = htools.x2hdm(param, X)    
        for m in range(M1):
            index_set = np.random.choice(N_choose_2_2, K*N_choose_2,replace=False)
            index_m = index[index_set,:]
            output = hdm.HDM_order_sensitivity(param, D, index_m)
            Dt[i,m,:,:] = htools._arccosh(output.G)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(directory+'/D'+'_N_'+str(N)+'_S_'+str(S)+'.npy', Dt)

def odor_embedding(param,K):
    directory = param.path
    if not os.path.exists(directory):
        os.makedirs(directory)

    name = 'blueberry'
    C,label = htools.read_odor(name)
    N = np.shape(C)[1]
    param.N = N
    D = -C
    index = htools.ordinal_set(param)
    N_choose_2 = int(N*(N-1)/2)
    N_choose_2_2 = int( N_choose_2 * ( N_choose_2-1)/2 )
    S = 1-K*N_choose_2/N_choose_2_2
    index_set = np.random.choice(N_choose_2_2, K*N_choose_2,replace=False)
    index_m = index[index_set,:]

    output = hdm.EDM_order(param, D, index_m)
    G = output.G
    np.save(directory+'/G_EDM.npy', G)
    for d in range(5):
        param.d = 2*(d+1)
        Gd = htools.rankProj(G, param)
        D = htools.gram2edm(Gd, param, False)
        r = htools.recontruction_accuracy(param,D,-C)
        print('Odor embedding accuracy', r, 'for d=',param.d,'and for EDM')
    
    output = hdm.HDM_order(param, D, index_m)
    G = output.G
    np.save(directory+'/G_HDM.npy', G)
    for d in range(5):
        param.d = 2*(d+1)
        Gd = htools.h_rankPrj(htools.valid(G), param)
        D = htools._arccosh(Gd)
        r = htools.recontruction_accuracy(param,D,-C)
        print('Odor embedding accuracy', r, 'for d=',param.d,'and for HDM')
    
