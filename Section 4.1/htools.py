#!/usr/bin/env python
# coding=utf-8
###########################################################################
import numpy as np
import cvxpy as cvx
import random
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
        print('problem in x2hgram 111')
    #######################################
    E2 = G-l_rankPrj(param, G)
    if np.linalg.norm(E2,'fro') > 1e-10:
        print(np.linalg.norm(E2,'fro'))
        print('problem in x2hgram 222')
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
def x2hdm(param, X):
    G = x2hgram(param, X)
    D = _arccosh(G)
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
def randX(param):
    N = param.N
    d = param.d
    #######################################
    X = np.random.normal(0, np.sqrt(N), (d+1,N))
    for n in range(N):
        X[0,n] = np.sqrt(1+np.linalg.norm(X[1:d+1,n])**2);
    return X
###########################################################################
def sym_noise_normal(param):
    N = param.N
    noise = np.random.randn(N,N)
    noise = ( 2 ** (-1/2) ) * (noise + noise.T)
    return noise
###########################################################################
def sym_noise(param):
    std = param.std
    #######################################
    noise = std*sym_noise_normal(param)
    np.fill_diagonal(noise, 0)
    return noise
###########################################################################
def generateData(param):
    X = randX(param)
    D = x2hdm(param, X)
    noise = sym_noise(param)
    W = randMask(param)
    #######################################
    Dn = D + noise
    #######################################
    e = np.linalg.norm(D-Dn,'fro') / np.linalg.norm(D,'fro') 
    return X, D, Dn, W, e
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
def x2edm(param, X):
    G = np.matmul(X.T,X)
    D = gram2edm(G, param, False)
    return D
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
        print('rank projection problem')
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
def error_ratio(D, Dn):
    return np.linalg.norm(Dn-D,'fro') / np.linalg.norm(D,'fro') 
###########################################################################
def test_error(param, D, Gn, mode,W):
    if mode == 1:
        Dn = _arccosh(Gn)
    else:
        Dn = np.sqrt( gram2edm(Gn, param, False) )
    eo = np.linalg.norm(np.multiply(W,Dn-D),'fro') / np.linalg.norm(np.multiply(W,D),'fro') 
    return eo
###########################################################################
def tree(grandparents, parent, children, adjM, D):
        M = len(children)
        for i in range(M):
            w = np.random.uniform(0,1,1)
            c_i = children[i]
            adjM[parent, c_i] = 1
            adjM[c_i, parent] = 1
            D[parent, c_i] = w
            D[c_i, parent] = w
            for j in range(len(grandparents)):
                g_j = grandparents[j]
                D[g_j,c_i] = D[g_j,parent]+D[parent, c_i]
                D[c_i,g_j]=D[g_j,c_i]
        for i in range(M):
            c_i = children[i]
            for j in range(M):
                c_j = children[j]
                if i != j:
                    D[c_j, c_i] = D[parent, c_i] + D[parent, c_j]
                    D[c_i, c_j] = D[c_j, c_i]
        return adjM, D
###########################################################################
def randomTree(param, max_degree):
    N = param.N
    adjM = np.zeros((N,N))
    D = np.zeros((N,N))
    length = 1
    node_list = np.arange(0,N)
    grand_parents = node_list[0:0]
    child_list = node_list
    child_list = np.delete(child_list, 0)
    flag = False
    length = 1
    for n in range(N):
        parent = node_list[n:n+1]
        grand_parents = np.append(grand_parents, parent)
        i_max = random.randint(1,max_degree)
        #max_degree + n//max_degree #
        if length > N:
            break
        if length + i_max > N:
            i_max = N - length 
            flag = True
        length = length + i_max 
        children = child_list[0:i_max]
        child_list = np.delete(child_list, range(i_max), None)
        adjM, D = tree(grand_parents, parent, children, adjM, D)
        grand_parents = np.append(grand_parents, children)
        if flag:
            break
    return D, adjM
###########################################################################
def randomRbRu(param):
    d = param.d
    #######################################
    b = np.random.randn(d,1)
    #######################################
    [U,S,V] = np.linalg.svd(np.random.randn(d,d),full_matrices=True)
    #######################################
    Rb, Ru = RbRu(param, U, b)
    return Rb, Ru, b[:,0]
###########################################################################
def RbRu(param, U, b):
    d = param.d
    #######################################
    Ru = np.zeros([d+1,d+1])
    Rb = np.zeros([d+1,d+1])
    #######################################
    Ru[0,0] = 1
    Ru[1:d+1,1:d+1] = U
    #######################################
    b_norm = np.linalg.norm(b)
    Rb[0,0] = np.sqrt(1+b_norm**2)
    Rb[0,1:d+1] = b.T
    Rb[1:d+1,0] = b[:,0]
    bbi = np.eye(d)+np.matmul(b,b.T)
    [U,S,V] = np.linalg.svd(bbi,full_matrices=True)
    S = np.diag(S)
    S_sq = np.sqrt(S)
    bbi_sq = np.matmul(U,np.matmul(S_sq,U.T))
    Rb[1:d+1,1:d+1] = bbi_sq
    #######################################
    return Rb, Ru
###########################################################################
def edgeCnt(param):
    N = param.N
    #######################################
    if param.bipartite:
        N0 = param.N0
        N1 = N-N0
        M = N0*N1
    else:
        M = N*(N-1)//2
    return M
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
        print('problem in h_rankProj 111')
    if np.linalg.norm(G_-l_rankPrj(param, G_),'fro')/N > 1e-7:
        #print(np.linalg.norm(G_-l_rankPrj(param, G_),'fro'))
        print('problem in h_rankProj 222')
    #######################################
    return G_
###########################################################################
def find_index(M,N):
    count = 1
    for i in range(N):
        for j in range(i+1, N):
            if count == M:
                return i,j
            else:
                count = count +1
###########################################################################
def next_index(i,j,N):
    if j < N-1:
        return i, j+1
    if j == N-1:
        return i+1,i+2
###########################################################################
def read_mnist(M):
	X = np.zeros([28*28,10*M])
	lab = np.zeros(10*M)
	import cv2
	import matplotlib.pyplot as plt
	for i in range(10):
		address = 'MNIST/'+str(i)+'/img_'
		for j in range(M):
			#print(address+str(j+1)+'.jpg')
			img = cv2.imread(address+str(j+1)+'.jpg',0)
			img = np.reshape(img, 28*28)
			X[:,j+M*i] = img
			lab[j+M*i] = i
	return X, lab

def e(i,n):
	ei = np.zeros([int(n),1])
	ei[int(i),0] = 1
	return ei
	#print(img)
	#cv2.imshow('image',X)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	#img = cv2.imread('/MNIST/0/img_1.jpg')
    #image = plt.imread('/MNIST/0/img_1.jpg')
	#image = Image.open("image_path.jpg")
	#img=Image.open("git/PhD/distance-geometry-pubs/hdm//database/MNIST/0/img_1.jpg")
	#imgplot = plt.imshow(img)
###########################################################################
def min_max_dist_odor(C):
    N = np.shape(C)[0]
    np.fill_diagonal(C, 0)
    #print(np.argmax(C))
    i,j = np.unravel_index(C.argmax(), C.shape)
    k,l = np.unravel_index((-C).argmax(), C.shape)
    return i,j, k,l

def psd_approx(G):
    w,v = np.linalg.eig(G)
    #print(w)
    w=w.real
    w[w<0] = 0
    v = v.real
    G0 = np.matmul(np.matmul(v,np.diag(w)),v.T) 
    G0 = (G0+G0.T)/2
    return G0

def sqrt(G):
    w,v = np.linalg.eig(G)
    w=w.real
    w[w<0] = 0
    w = np.sqrt(w)
    v = v.real
    G0 = np.matmul(np.matmul(v,np.diag(w)),v.T)
    return G0


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


def recontruction_accuracy(param,D1,D2):
    index = ordinal_set(param)
    M = np.shape(index)[0]
    count = 0
    count_correct = 0
    for m in M:
        i = index[m,0]
        j = index[m,1]
        k = index[m,2]
        l = index[m,3]
        if D1[i,j] <= D1[k,l] and D2[i,j] <= D2[k,l]:
            count_correct = count_correct+1
        if D1[i,j] >= D1[k,l] and D2[i,j] >= D2[k,l]:
            count_correct = count_correct+1
        count = count +1
    r = count_correct/count
    return r
        

def P(G,d,N):
    #[U,S,V] = np.linalg.svd(G,full_matrices=True)
    w,v = np.linalg.eig(G)
    v = v.real
    v_ = np.linalg.inv(v)
    #print(np.linalg.norm(np.matmul(np.matmul(v,np.diag(w)),v_)-G,'fro'))
    #print(np.linalg.norm(np.matmul(np.matmul(U,np.diag(S)),V)-G,'fro'))
    #print(w)
    
    w = w.real
    ind = np.argsort(-w)
    ind = ind[d:N]
    v = v[:,ind]
    print('diagonal elements of v.T times v is', np.diag(np.matmul(v.T,v)))
    v_ = v_[ind,:]
    #print(S)
    #u = U[:,d:N]
    #v = V[d:N,:]
    G0 = np.matmul(v,v.T)
    #G0 = G0.T
    return G0


def read_odor(name):
    if name == 'blueberry':
        C = np.load('odor/C_blue.npy')
        lab = np.load('odor/label_blue.npy')
        print('blueberries!!')
    if name == 'strawberry':
        C = np.load('odor/C_straw.npy')
        lab = np.load('odor/label_straw.npy')
        print('strawberries!!')
    if name == 'tomato':
        C = np.load('odor/C_tomato.npy')
        lab = np.load('/odor/label_tomato.npy')
        print('tomatos!!!')
    return C, lab