import cvxpy as cvx
import numpy as np
import htools
###########################################################################
class HDM_OUT:
    def __init__(self):
        self.x          = None
        self.D          = None
        self.r          = None
        self.status     = None
###########################################################################
class EDM_OUT:
    def __init__(self):
        self.x          = None
        self.D          = None
        self.r          = None
        self.status     = None
###########################################################################
def EDM_order(param, D, index):
    N = param.N
    d = param.d
    M = np.shape(index)[0]
    #######################################
    output = HDM_OUT()
    gamma = 0.05
    EPS = param.eps
    p = param.p
    #######################################
    G = cvx.Variable((N, N), PSD=True)
    x = cvx.Variable(M)
    one = np.ones((N,1))
    #######################################
    con = []
    con.append(G*one == 0)
    con.append(x >= 0)
    con.append(cvx.sum(x) <= EPS*p*M)
    #######################################
    for m in range(M):
        i1 = int(index[m,0])
        i2 = int(index[m,1])
        i3 = int(index[m,2])
        i4 = int(index[m,3])
        if D[i1,i2] < D[i3,i4]:
            con.append(G[i1,i1]+G[i2,i2]-2*G[i1,i2] <= G[i3,i3]+G[i4,i4]-2*G[i3,i4] - EPS+x[m])
        if D[i1,i2] > D[i3,i4]:
            con.append(G[i1,i1]+G[i2,i2]-2*G[i1,i2] >= G[i3,i3]+G[i4,i4]-2*G[i3,i4] + EPS-x[m])
    #######################################
    min_dist = 1
    k,l = htools.min_dist(D)
    con.append(-2*G[k,l]+G[k,k]+G[l,l] == min_dist)
    for i in range(N):
        for j in range(i+1,N):
            if i == k and j == l:
                continue
            if i == l and j == k:
                continue
            con.append(-2*G[i,j]+G[i,i]+G[j,j] >= min_dist)
    #######################################
    cost = cvx.trace(G)+(gamma/EPS)*cvx.sum(x)
    #######################################
    obj = cvx.Minimize(cost)
    prob = cvx.Problem(obj,con)
    try:
        prob.solve(solver=cvx.CVXOPT,verbose=True,normalize = False,#, #max_iter= 500, 
            kktsolver = 'robust')#, refinement = 2)
    except Exception as message:
        print(message)
    output.status = str(prob.status)
    if str(prob.status) == 'optimal':
        Gd = htools.rankProj(G.value, param)
        Dr = htools.gram2edm(Gd, param, False)
        output.D        = Dr
        output.r        = htools.recontruction_accuracy(param,Dr,D)
        output.x        = x.value
    else:
        print('fail')
    return output
###########################################################################
def HDM_order(param, D, index):
    N = param.N
    d = param.d
    M = np.shape(index)[0]
    #######################################
    output = HDM_OUT()
    gamma = 0.1
    EPS = param.eps
    p = param.p
    #######################################
    Gp = cvx.Variable((N, N), PSD=True)
    Gn = cvx.Variable((N, N), PSD=True)
    x = cvx.Variable(M)
    G = Gp-Gn
    #######################################
    con = []
    con.append(cvx.diag(G) == -1)
    con.append(x >= 0)
    con.append(cvx.sum(x) <= EPS*p*M)
    #######################################
    for m in range(M):
        i1 = int(index[m,0])
        i2 = int(index[m,1])
        i3 = int(index[m,2])
        i4 = int(index[m,3])
        if D[i1,i2] < D[i3,i4]:
            con.append(G[i1,i2] - G[i3,i4] >= EPS-x[m])
        if D[i1,i2] > D[i3,i4]:
            con.append(G[i3,i4]- G[i1,i2] >= EPS-x[m])
    #######################################
    min_dist = 1
    k,l = htools.min_dist(D)
    con.append(G[k,l] == htools._cosh(min_dist))
    for i in range(N):
        for j in range(i+1,N):
            if i == k and j == l:
                continue
            if i == l and j == k:
                continue
            con.append(G[i,j] <= htools._cosh(min_dist))
    #######################################
    cost = cvx.trace(Gn)+cvx.trace(Gp)+(gamma/EPS)*cvx.sum(x)
    #######################################
    obj = cvx.Minimize(cost)
    prob = cvx.Problem(obj,con)
    try:
        prob.solve(solver=cvx.CVXOPT,verbose=True,normalize = False, #max_iter= 500, 
            kktsolver = 'robust')#, refinement = 2)
    except Exception as message:
        print(message)
    output.status = str(prob.status)
    if str(prob.status) == 'optimal':
        Gd = htools.h_rankPrj(htools.valid(G.value), param)
        Dr = htools._arccosh(Gd)
        output.D        = Dr
        output.r        = htools.recontruction_accuracy(param,Dr,D)
        output.x        = x.value
    else:
        print('fail')
    return output
###########################################################################
def odor_embedding(C,param):
    #######################################
    N = param.N
    K = param.K
    mode = param.mode
    #######################################
    D = -C
    #######################################
    index = htools.ordinal_set(param)
    N_choose_2 = int(N*(N-1)/2)
    N_choose_2_2 = int( N_choose_2 * ( N_choose_2-1)/2 )
    #######################################
    if mode == 'Euclidean':
        #######################################
        index_set = np.random.choice(N_choose_2_2, int(K*N_choose_2),replace=False)
        index_m = index[index_set,:]
        #######################################
        output = EDM_order(param, D, index_m)
    else:
        #######################################
        index_set = np.random.choice(N_choose_2_2, int(2*K*N_choose_2),replace=False)
        index_m = index[index_set,:]
        #######################################
        output = HDM_order(param, D, index_m)
        #######################################
    return output
###########################################################################