#!/usr/bin/env python
# coding=utf-8
###########################################################################
import numpy as np
import cvxpy as cvx
###########################################################################
# Returns the mean of the vector V
def mean(V):
    N = len(V)
    Vm = 0
    for n in range(N):
        Vm += V[n]
    return Vm/N
###########################################################################
# Returns K
def K_base(param):
    P = param.P
    mode = param.mode
    if mode == 1:
        K = 2*P+1
    else:
        K = 4*P+1
    return K
###########################################################################
def randomAs(param):
    A = []
    N = param.N
    d = param.d
    P = param.P
    mode = param.mode
    if mode == 1:
        for p in range(P+1):
            A.append(np.random.randn(d,N))
    else:
        j = (-1) ** (1/2)
        for p in range(P):
            Ar = np.random.randn(d,N)
            Aq = np.random.randn(d,N)
            A.append(Ar+j*Aq)
        A.append(np.random.randn(d,N))
        for p in range(P):
            A.append(np.conj(A[P-p-1]))
    return A
###########################################################################
# Returns X(t)
def X_t(A,t,param):
    N = param.N
    d = param.d
    P = param.P
    omega = param.omega
    mode = param.mode
    j = (-1) ** (1/2)
    X = np.zeros((d,N))+0*j
    if mode == 1:
        for p in range(P+1):
            X += np.multiply(t**p, A[p])
    else:
        for p in range(P+1):
            ci = np.exp(j*p*omega*t)
            X += np.multiply(ci,A[p+P])+np.multiply(np.conj(ci),A[P-p])
        X -= A[P]
    X = np.real(X)
    return X
###########################################################################
# Generate random masks and returns data samples with their corresponding relative error
def generateData(param,trn_list,A):
    N = param.N
    N_trn = len(trn_list)
    std = param.std
    
    D = np.zeros((N_trn,N,N))
    W = np.zeros((N_trn,N,N))
    ei = np.zeros(N_trn)
    
    for i_t, t in enumerate(trn_list):
        Xi = X_t(A,t,param)
        Gi = np.matmul(Xi.T,Xi)
        Di = gram2edm(Gi, N, False)
        D[i_t] = projD(Di + std*symNoise(N))
        W[i_t] = randMask(param)
        ei[i_t] = np.linalg.norm(Di-D[i_t],'fro') / (np.linalg.norm(Di,'fro') )
    return D, W, ei
###########################################################################
def gram2edm(G, N, mode):
    if mode:
        dG = cvx.vstack( cvx.diag(G) )
        D = cvx.matmul(dG ,np.ones((1,N)))-2*G + cvx.matmul(np.ones((N,1)),dG.T)
    else:
        dG = np.diag(G)[:,None]
        D = dG-2*G+dG.T
    return D
###########################################################################
def symNoise(N):
    Noise = np.random.randn(N,N)
    Noise = ( 2 ** (-1/2) ) * (Noise + Noise.T)
    return Noise
###########################################################################
def projD(D):
    D = D - np.diag(np.diag(D))
    D[D <= 0.0] = 0.0
    return D
###########################################################################
def randMask(param):
    N = param.N
    W = np.zeros((N,N))
    n_del = param.n_del
    if param.bipartite:
        N0 = param.N0
        N1 = N - N0
        M = edgeCnt(param)
        k = np.random.choice(np.arange(M),size=(n_del), replace = False)
        Wb = np.ones((N0,N1))
        ij = np.where(Wb)
        for i in k:
            Wb[ij[0][i], ij[1][i]] = 0
        W[0:N0,N-N1:] = Wb
        W = W + W.T
    else:
        M = edgeCnt(param)
        k = np.random.choice(np.arange(M),size=(n_del), replace = False)
        ij = np.where(np.tril(np.ones((N,N)), k=-1))
        for i in k:
            W[ij[0][i], ij[1][i]] = 1
        W = 1 - (W+W.T)
    return W
###########################################################################
def generalsampler(param, sampler):
    import math
    K = param.K
    mode = param.mode
    omega = param.omega
    sampling = param.sampling
    T0 = 2*np.pi/omega
    ################################################################
    if sampler == 'basis':
        N = K
        T = param.T_tst
    elif sampler == 'trn_sample':
        N = param.N_trn
        T = param.T_trn
    elif sampler == 'tst_sample':
        N = param.N_tst
        T = param.T_tst
    else:
        N = param.Nr
        T = np.array([0,45])
    ################################################################
    if mode == 2:
        T[1] = min(T[1]-T[0],T0)+T[0]
    ################################################################
    if sampler == 'basis' or sampler == 'tst_sample':
        if mode == 1:
            interval = np.linspace(T[0],T[1],N)
        else:
            interval = np.linspace(T[0],T[1],N+1)
            interval = interval[0:N]
    elif sampler == 'trn_sample':
        if mode == 1:
            interval = np.linspace(T[0],T[1],N)
        else:
            interval = np.linspace(T[0],T[1],N+1)
            interval = interval[0:N]
        if sampling == 2:
            if mode == 1:
                for i in range(N):
                    interval[i] = (T[0]+T[1])/2 + (T[1]-T[0])/2 * math.cos((2*i+1)/(2*N) * np.pi)
            else:
                interval = np.linspace(T[0],T[1],N+1)
                for i in range(N+1):
                    interval[i] = (T[0]+T[1])/2 + (T[1]-T[0])/2 * math.cos((2*i+1)/(2*(N+1)) * np.pi)
                interval = interval[0:N]
        elif sampling == 3:
            interval = np.random.uniform(T[0],T[1],N)
    else:
        if mode == 1:
            Nc = int(np.ceil(N/2))
            Nf = int(np.floor(N/2))
            interval_p = np.exp(np.random.uniform(T[0],T[1],Nc)-5) - np.exp(-5)
            interval_n = -np.exp(np.random.uniform(T[0],T[1],Nf)-5) + np.exp(-5)
            interval = np.concatenate((interval_n, interval_p), axis=0)
        else:
            interval = np.random.uniform(T[0],T[1],N)

    return interval
###########################################################################
def G_t(G, w, mode):
    K = np.shape(w)[0]
    if mode:
        G_tot = w[0]*G[0]
        for k in range(K-1):
            G_tot += w[k+1]*G[k+1]
    else:
        G_tot = np.zeros(G[0].shape)
        for k in range(K):
            G_tot += np.multiply(w[k],G[k])
    return G_tot
###########################################################################
def gram2x(G,d):
    N = G.shape[0]
    [U,S,V] = np.linalg.svd(G,full_matrices=True)
    S = S ** (1/2)
    S = S[0:d]
    X = np.matmul(np.diag(S),V[0:d])
    return X
###########################################################################
def rankProj(G,param):
    N = param.N
    d = param.d
    K = param.K
    for k in range(K):
        [U,S,V] = np.linalg.svd(G[k].value,full_matrices=True)
        S[d:N] = 0
        G[k] = np.matmul(np.matmul(U,np.diag(S)),V)
    return G
###########################################################################
def W(t, tau_list, param):
    P = param.P
    mode = param.mode
    omega = param.omega
    K = param.K

    M = np.zeros((K,K))
    T = np.zeros((K,1))
    if mode == 1:
        for i in range(K):
            M[i,:] = tau_list**i
            T[i,0] = t**i
    else:
        for i in range(K):
            if i % 2 == 1:
                f = (i+1)/2
                M[i,:] = np.sin(omega*f*tau_list)
                T[i,0] = np.sin(omega*f*t)
            else:
                f = i/2
                M[i,:] = np.cos(omega*f*tau_list)
                T[i,0] = np.cos(omega*f*t)
    weights = np.matmul(np.linalg.inv(M), T)
    return weights
###########################################################################
def edgeCnt(param):
    N = param.N
    if param.bipartite:
        N0 = param.N0
        N1 = N-N0
        M = N0*N1
    else:
        M = N*(N-1)//2
    return M
###########################################################################
def fourier2sine(A):
    L, d, N = np.shape(A)
    C = []
    P = (L-1)//2
    j = (-1)**(1/2)
    
    C.append(np.real(A[P]))
    for i in range(2*P):
        p = i//2+1
        if i % 2 == 1:
            C.append( np.real(A[p+P]+np.conj(A[p+P])) )
        else:
            C.append( np.real(j*(A[p+P]-np.conj(A[p+P]))) )
    return C
###########################################################################
def sine2fourier(A):
    L, d, N = np.shape(A)
    C = []
    P = (L-1)//2
    j = (-1)**(1/2)
    
    for p in range(-P,P+1):
        C.append(np.zeros((d,N)))
    C[P] = A[0]
    for p in range(P):
        C[P+p+1] = (-j*A[2*p+1]+A[2*p+2])/2
        C[P-p-1] = np.conj(C[P+p+1])
    return C
###########################################################################
def T_t(t, param):
    d = param.d
    P = param.P
    omega = param.omega
    mode = param.mode
    eyed = np.eye(d)
    if mode == 1:
        for p in range(P+1):
            Tp = (t**p)*eyed
            if p == 0:
                T = Tp
            else:
                T = np.concatenate((T, Tp), axis=1)
    else:
        for i in range(2*P+1):
            if i % 2 == 1:
                f = (i+1)/2
                Tp = np.sin(omega*f*t)*eyed
            else:
                f = i/2
                Tp = np.cos(omega*f*t)*eyed
            if i == 0:
                T = Tp
            else:
                T = np.concatenate((T, Tp), axis=1)
    return T
###########################################################################
def deconcat(A_, param):
    d = param.d
    P = param.P
    mode = param.mode
    A = []
    if mode == 1:
        for i in range(P+1):
            A.append(A_[i*d:(i+1)*d,:])
    else:
        for i in range(2*P+1):
            A.append(A_[i*d:(i+1)*d,:])
    return A
###########################################################################
def testError(param, tau_list, tst_list, G, A):
    N = param.N
    P = param.P
    omega = param.omega
    mode = param.mode
    error = 0
    snr = 0
    N_tst = len(tst_list)
    eo = np.zeros(N_tst)
    for i_t, t in enumerate(tst_list):
        weights = W(t,tau_list,param)
        G_tot = G_t(G,weights,False)
        D_G = gram2edm(G_tot,N,False)
        X = X_t(A,t,param)
        eo[i_t] = np.linalg.norm(edm(X)-D_G,'fro') / np.linalg.norm(edm(X),'fro')
    return eo
###########################################################################
def rotationXY(X,Y): ## X is anchor, Y is not
    N = X.shape[1]
    J = np.eye(N) - np.ones((N,N))/N
    XY = np.matmul(np.matmul(X,J),np.conjugate(Y.T))
    U,_,V = np.linalg.svd(XY,full_matrices = True)  ## XJY' = UV' = R
    R = np.matmul(U,V)
    return R
###########################################################################
def randomAnchor(M,N,n):
    # N : Number of all points
    # M : Time samples
    # n : Number of anchors
    anchor_idx = np.zeros((M,N))
    idx = np.zeros(N)
    idx[0:n] = 1
    for m in range(M):
        np.random.shuffle(idx)
        anchor_idx[m,:] = idx
    return anchor_idx
###########################################################################
def satTrj(param):
    N = param.N
    P = param.P
    N_list = np.random.uniform(0,1,P)
    N_list = N_list / np.sum(N_list)
    
    if P==1:
        N_list[P-1] = N
    else:
        N_list = N*N_list
        N_list = np.floor(N_list)
        N_list[P-1] = N - np.sum(N_list[0:P-1])

    L = len(N_list)
    rad = [None]*L
    for l in range(L):
        N_l = int(N_list[l])
        rad[l] = np.exp(np.random.uniform(1,2,(2,N_l) ))

    rot = [None]*N
    for n in range(N):
        X = np.random.randn(3,3)
        [U,S,V] = np.linalg.svd(X,full_matrices=True)
        rot[n] = np.matmul(U,V.T)

    A = np.zeros((2*P+1,3,N))
    A[0,:,:] = np.zeros((3,N))
    for l in range(L):
        N_l = int(N_list[l])
        r = rad[l]
        if l==0:
            N_left = int(0)
        else:
            N_left = int(np.sum(N_list[0:l]))
        TMP = np.zeros((3,N))
        TMP[0,N_left:N_left+N_l] = r[0,:]
        for n in range(N):
            TMP[:,n] = np.matmul(rot[n],TMP[:,n])
        A[2*l+1,:,:] = TMP
        TMP = np.zeros((3,N))
        TMP[1,N_left:N_left+N_l] = r[1,:]
        for n in range(N):
            TMP[:,n] = np.matmul(rot[n],TMP[:,n])
        A[2*l+2,:,:] = TMP
    A = sine2fourier(A)
    return A
###########################################################################
def edm(X, Y=None):
    if Y is None:
        Y = X
    
    d = X.shape[0]
    m = X.shape[1]
    n = Y.shape[1]
    D = np.zeros((m, n))

    for i in range(d):
        D += (X[np.newaxis, i, :].T - Y[i, :])**2
    return D
