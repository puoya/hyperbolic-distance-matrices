#!/usr/bin/env python
# coding=utf-8
#import hdm
#import htools
import numpy as np
import os
import htools
import hdm
def tree_embedding(param,M):
    mode = param.space
    N = param.N
    max_degree = 3
    directory = param.path
    if not os.path.exists(directory):
        os.makedirs(directory)
    for m in range(M):
        D, adjM = htools.randomTree(param, max_degree)
        np.save(directory+'/D'+'_N_'+str(N)+'_M_'+str(m)+'.npy', D)
        np.save(directory+'/M'+'_N_'+str(N)+'_M_'+str(m)+'.npy', adjM)
    if not os.path.exists(directory+'/'+mode):
        os.makedirs(directory+'/'+mode)
    if mode == 'Hyperbolic':
        for m in range(M):
            print('N = ', N,',m = ',m,' and ', mode, 'space')
            D = np.load(directory+'/D'+'_N_'+str(N)+'_M_'+str(m)+'.npy') 
            output = hdm.HDM_metric(param,D, D, np.ones((N,N)),param.cost,param.norm)
            G = htools.valid( output.G )
            np.save(directory+'/'+mode+'/G'+'_N_'+str(N)+'_M_'+str(m)+'.npy', G)
            e_rel = 0
            D0 = htools._arccosh(G)
            D0_norm = np.linalg.norm(D0,'fro')   
            for d in range(2,N):
                param.d = d
                Gd = htools.h_rankPrj(G, param)
                Gd = htools.valid(Gd)
                Dd = htools._arccosh(Gd)
                error_d = np.linalg.norm(D0-Dd,'fro') / D0_norm
                if abs(error_d - e_rel) < 1e-3:
                    break
                e_rel = error_d
            np.save(directory+'/'+mode+'/d'+'_N_'+str(N)+'_M_'+str(m)+'.npy', d)
            param.d = d
            Gd = htools.h_rankPrj(G, param)
            Gd = htools.valid(Gd)
            Dd = htools._arccosh(Gd)
            error_d_0 = np.linalg.norm(D-Dd,'fro') / np.linalg.norm(D,'fro')   
            np.save(directory+'/'+mode+'/e'+'_N_'+str(N)+'_M_'+str(m)+'.npy', error_d_0)
    else:
        for m in range(M):
            print('N = ', N,',m = ',m,' and mode is', mode)
            D = np.load(directory+'/D'+'_N_'+str(N)+'_M_'+str(m)+'.npy') 
            output = hdm.EDM_metric(param,D, D, np.ones((N,N)))
            G = output.G
            np.save(directory+'/'+mode+'/G'+'_N_'+str(N)+'_M_'+str(m)+'.npy', G)
            e_rel = 0

            D0 = htools.gram2edm(G, param, False)
            D0_norm = np.linalg.norm(D0,'fro')   
            for d in range(2,N):
                param.d = d
                Gd = htools.rankProj(G, param)
                Dd = htools.gram2edm(Gd, param, False)
                error_d = np.linalg.norm(D0-Dd,'fro') / D0_norm
                if abs(error_d - e_rel) < 1e-3:
                    break
                e_rel = error_d
            np.save(directory+'/'+mode+'/d'+'_N_'+str(N)+'_M_'+str(m)+'.npy', d)
            param.d = d
            Gd = htools.rankProj(G, param)
            Dd = htools.gram2edm(Gd, param, False)
            error_d_0 = np.linalg.norm(D-Dd,'fro') / np.linalg.norm(D,'fro')   
            np.save(directory+'/'+mode+'/e'+'_N_'+str(N)+'_M_'+str(m)+'.npy', error_d_0)