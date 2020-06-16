#!/usr/bin/env python
# coding=utf-8
###########################################################################
import numpy as np
import cvxpy as cvx
import random
from numpy import linalg as LA
###########################################################################
def cross_correlation(R):
    M,N = np.shape(R)
    mu = R.mean(axis = 0)
    mu = np.asmatrix(mu)
    R = R-np.matmul(np.ones((M,1)),mu)
    for n in range(N):
        R[:,n] = R[:,n]/LA.norm(R[:,n])
    C = np.matmul(R.T,R)
    return C
###########################################################################
def h_norm(x, param):
    d = param.d
    #######################################
    H = np.eye(d+1)
    H[0,0] = -1
    #######################################
    x_norm = np.matmul(np.matmul(x.T,H),x)
    return x_norm
###########################################################################
def x2lgram(param, X):
    d = param.d
    #######################################
    H = np.eye(d+1)
    H[0,0] = -1
    #######################################
    G = np.matmul(np.matmul(X.T,H),X)
    return G
###########################################################################
def x2hgram(param, X):
    N = param.N
    #######################################
    X_ = X
    for n in range(N):
        x = X_[:,n]
        X_[:,n] = projectX(param, x)
    #######################################
    G = x2lgram(param, X_)
    #######################################
    E1 = G-valid(G)
    if np.linalg.norm(E1,'fro') > 1e-10:
        print(np.linalg.norm(E1,'fro'))
        print('inaccuracy in htools.x2hgram - ')
    #######################################
    E2 = G-l_rankPrj(param, G)
    if np.linalg.norm(E2,'fro') > 1e-10:
        print(np.linalg.norm(E2,'fro'))
        print('inaccuracy in htools.x2hgram -- ')
    return G
###########################################################################
def valid(G):
    np.fill_diagonal(G, -1)
    G[G >= -1] = -1
    return G
###########################################################################
def lgram2x(param, G):
    #######################################
    d = param.d
    #######################################
    w, v = np.linalg.eig(G)
    w = w.real
    v = v.real
    lambda_0 = np.amin(w)
    ind_0 = np.argmin(w)
    w = np.delete(w, ind_0)
    ind = np.argsort(-w)
    w = -np.sort(-w)
    ind = ind[:d]
    w = w[:d]
    #######################################
    lambda_ = np.concatenate((abs(lambda_0), w), axis=None) 
    lambda_[lambda_ <= 0] = 0
    lambda_ = np.sqrt(lambda_)

    ind_ = np.concatenate((ind_0, ind+1), axis=None) 
    v = v[:, ind_]
    X = np.matmul(np.diag(lambda_),v.T)
    #######################################
    if X[0,0] < 0:
        X = -X
    #######################################
    return X  
###########################################################################
def hgram2x(param, G):
    N = param.N
    #######################################
    X = lgram2x(param, G)
    #######################################
    for n in range(N):
        x = X[:,n]
        X[:,n] = projectX(param,x)
    #######################################
    return X  
###########################################################################
def _arccosh(G):
    D = np.arccosh(-G)
    return D
###########################################################################
def _cosh(D):
    l = np.where(np.isnan(D))
    for i in range(len(l[0])):
        print('NaN in HDM')
        D[l[0][i],l[1][i]] = 1e200
    G = -np.cosh(D)
    return G
###########################################################################
def l_rankPrj(param, G):
    X = lgram2x(param, G)
    return x2lgram(param, X)
###########################################################################
def projectX(param,x):
    d = param.d
    #######################################
    H = np.eye(d+1)
    H[0,0] = -1
    I = np.eye(d+1)
    #######################################
    x0 = x[0]
    #print('x0 is:', x[0])
    eps = 1e-15
    center = 0
    error = abs(h_norm(x, param)+1)
    #print(error)
    A_opt = I
    for i in range(50):
        l = 10**(-i)
        if x0 > 0:
            lambda_min = max(center-l, -1+eps)
            lambda_max = min(center+l, 1-eps)
            number = 50
        else:
            lambda_min = max(center-l*10000, 1+eps)
            lambda_max = center+l*10000
            number = 100
        lambda_list = np.linspace(lambda_min,lambda_max,num=number)
        for lambda_ in lambda_list:
            A = np.linalg.inv(I + lambda_*H)
            x_l = np.matmul(A,x)
            if abs(h_norm(x_l, param)+1) < error:
                A_opt = A
                error = abs(h_norm(x_l, param)+1)
                center = lambda_
    x_opt = np.matmul(A_opt,x)
    if error > 1e-5:
        print('hi:',error)
        print(center)
    return x_opt
###########################################################################
def gram2edm(G, param, mode):
    N = param.N
    if mode:
        dG = cvx.vstack( cvx.diag(G) )
        D = cvx.matmul(dG ,np.ones((1,N)))-2*G + cvx.matmul(np.ones((N,1)),dG.T)
    else:
        dG = np.diag(G)[:,None]
        D = dG-2*G+dG.T
        D[D <= 0] = 0
    return D
###########################################################################
def gram2x(G, param):
    N = param.N
    d = param.d
    [U,S,V] = np.linalg.svd(G,full_matrices=True)
    S = S ** (1/2)
    S = S[0:d]
    X = np.matmul(np.diag(S),V[0:d])
    return X
###########################################################################
def rankProj(G, param):
    N = param.N
    d = param.d
    [U,S,V] = np.linalg.svd(G,full_matrices=True)
    S[d:N] = 0
    G = np.matmul(np.matmul(U,np.diag(S)),V)
    if np.linalg.norm(G-G.T,'fro') > 1e-10:
        print('inaccuracy in htools.rankProj')
    return G
###########################################################################
def poincare2loid(Y):
    Y = Y.real
    d_,N = Y.shape
    X = np.zeros([d_+1,N])
    for n in range(N):
        y = Y[:,n]
        y_norm = np.linalg.norm(y)
        y = np.append(1+y_norm**2,2*y)
        y = y/(1-y_norm**2)
        X[:,n] = y
    return X
###########################################################################
def loid2poincare(X):
    X = X.real
    d_,N = X.shape
    Y = np.zeros([d_-1,N])
    if X[0,0] < 0:
        X = -X
    for n in range(N):
        x = X[:,n]
        x0 = x[0] ###########???????????????#################
        x = np.delete(x, 0)
        x = x/(x0+1)
        Y[:,n] = x
    return Y
###########################################################################
def h_rankPrj(G, param):
    N = param.N
    X = lgram2x(param, G)
    G_ = x2hgram(param, X)
    #######################################
    if np.linalg.norm(G_-valid(G_),'fro')/N > 1e-7:
        print('inaccuracy in htools.h_rankPrj - ')
    if np.linalg.norm(G_-l_rankPrj(param, G_),'fro')/N > 1e-7:
        #print(np.linalg.norm(G_-l_rankPrj(param, G_),'fro'))
        print('inaccuracy in htools.h_rankPrj -- ')
    #######################################
    return G_
###########################################################################
def next_index(i,j,N):
    if j < N-1:
        return i, j+1
    if j == N-1:
        return i+1,i+2
###########################################################################
def min_dist(D):
    N = np.shape(D)[0]
    np.fill_diagonal(D, 0)
    i,j = np.unravel_index(D.argmin(), D.shape)
    return i,j
###########################################################################
def ordinal_set(param):
    N = param.N
    N_choose_2 = int(N*(N-1)/2)
    N_choose_2_2 = int( N_choose_2 * ( N_choose_2-1)/2 )
    index = np.zeros([N_choose_2_2,4])
    count = 0
    i = 0
    j = 1
    for s0 in range(N_choose_2-1):
        k, l = next_index(i,j,N)
        for s1 in range(s0+1, N_choose_2):
            index[count,0] = i
            index[count,1] = j
            index[count,2] = k
            index[count,3] = l
            count = count +1
            k,l = next_index(k,l,N)
        i,j = next_index(i,j,N)
    return index
###########################################################################
def recontruction_accuracy(param,D1,D2):
    index = ordinal_set(param)
    M = np.shape(index)[0]
    count = 0
    count_correct = 0
    for m in range(M):
        i = int(index[m,0])
        j = int(index[m,1])
        k = int(index[m,2])
        l = int(index[m,3])
        if (D1[i,j] <= D1[k,l]) and (D2[i,j] <= D2[k,l]):
            count_correct = count_correct+1
        if D1[i,j] >= D1[k,l] and D2[i,j] >= D2[k,l]:
            count_correct = count_correct+1
        count = count +1
    r = count_correct/count
    return r
###########################################################################
def d2x(D,param):
    mode = param.mode
    N = param.N
    if mode =='Euclidean':
        J = np.eye(N) - np.ones((N,N))/N
        G = (-0.5)*np.matmul(J,np.matmul(D,J))
        X = gram2x(G, param)
    else:
        G = _cosh(D)
        X = hgram2x(param, G)
        X = loid2poincare(X)
    return X
###########################################################################
