import cvxpy as cvx
import numpy as np
import htools
import os
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
def EDM_metric(param,D, Dn, W):
    N = param.N
    d = param.d
    #######################################
    output      = EDM_OUT()
    #######################################
    con = []
    G = cvx.Variable((N, N), PSD=True)
    con.append(G == G.T)
    con.append(cvx.sum(G,axis = 0) == 0)
    Dg = htools.gram2edm(G, param, True)
    cost = cvx.norm( cvx.multiply(W, D-Dg),'fro' )**2
    #######################################
    obj = cvx.Minimize(cost)
    prob = cvx.Problem(obj,con)
    try:
        if param.solver == 'CVXOPT':
            prob.solve(solver=cvx.CVXOPT,verbose=False)
        else:
            prob.solve(solver=cvx.SCS,verbose=False)
    except Exception as message:
        print(message)
    output.status = str(prob.status)
    if str(prob.status) == 'optimal':
        output.G = G.value
    else:
        print(error, 'failed in Euclidean')
    return output
###########################################################################
def HDM_metric(param,D, Dn, W,mode, norm):
    N = param.N
    d = param.d
    #######################################
    output = HDM_OUT()
    G_n = htools._cosh(Dn)
    I = np.eye(N)
    #######################################
    error_list = param.error_list
    delta_list = param.delta_list
    for error in error_list: 
        if mode == 'TRACE':
            Gp = cvx.Variable((N, N), PSD=True)
            Gn = cvx.Variable((N, N), PSD=True)
            G = Gp-Gn
            #######################################
            con = []
            if norm == 'l1':
                con.append( cvx.norm( cvx.multiply(W,G-G_n),1) <= error*W.sum())
            elif norm == 'l2':
                con.append( cvx.norm( cvx.multiply(W,G-G_n),2 )**2 <= error)
            elif norm == 'p1':
                con.append( cvx.pnorm( cvx.multiply(W,G-G_n),1 ) <= error*W.sum())
            elif norm == 'fro':
                con.append( cvx.norm( cvx.multiply(W,G-G_n),'fro' )**2 <= error*W.sum())
            con.append(cvx.diag(G) == -1)
            con.append(G <= -1)
            cost = cvx.trace(Gn)+cvx.trace(Gp)
            #######################################
            obj = cvx.Minimize(cost)
            prob = cvx.Problem(obj,con)
            try:
                if param.solver == 'CVXOPT':
                    prob.solve(solver=cvx.CVXOPT,verbose=False)
                else:
                    prob.solve(solver=cvx.SCS,verbose=False)
            except Exception as message:
                print(message)
            output.status = str(prob.status)
            if str(prob.status) == 'optimal':
                output.G = Gp.value - Gn.value
                return output
            else:
                print(error, 'failed in TRACE mode')
        else:
            Wp = I
            Wn = I
            Gn_0 = I
            Gp_0 = I
            for delta in delta_list:
                #######################################
                Gp = cvx.Variable((N, N), PSD=True)
                Gp.value = Gp_0
                Gn = cvx.Variable((N, N), PSD=True)
                Gn.value = Gn_0
                G = Gp-Gn
                #######################################
                con = []
                if norm == '1':
                    con.append( cvx.norm( cvx.multiply(W,G-G_n),1) <= error*W.sum())
                elif norm == '2':
                    con.append( cvx.norm( cvx.multiply(W,G-G_n),2 )**2 <= error)
                elif norm == 'p1':
                    con.append( cvx.pnorm( cvx.multiply(W,G-G_n),1 ) <= error*W.sum())
                elif norm == 'fro':
                    con.append( cvx.norm( cvx.multiply(W,G-G_n),'fro' )**2 <= error*(N**2))
                con.append(cvx.diag(G) == -1)
                con.append(G <= -1)
                #######################################
                cost = cvx.trace(cvx.matmul(Wp,Gp))+cvx.trace(cvx.matmul(Wn,Gn))
                #######################################
                obj = cvx.Minimize(cost)
                prob = cvx.Problem(obj,con)
                try:
                    if param.solver == 'CVXOPT':
                        prob.solve(solver=cvx.CVXOPT,verbose=False, warm_start=True)
                    else:
                        prob.solve(solver=cvx.SCS,verbose=False, warm_start=True)
                except Exception as message:
                    print(message)
                output.status = str(prob.status)
                if str(prob.status) == 'optimal':
                    Gp_0 = htools.psd_approx(Gp.value)
                    Gn_0 = htools.psd_approx(Gn.value)
                    #######################################
                    Wn = np.linalg.pinv(Gn_0 + (delta/d)*I)
                    Wp = np.linalg.pinv(Gp_0 + delta*I)
                    #######################################
                    output.G = Gp_0 - Gn_0
                    if delta == delta_list[-1]:
                        return output
                else:
                    print(error, 'failed in LOG_DET mode')
                    break
    return output
###########################################################################
def FindMaxSprs(param):
    maxIter = param.maxIter
    delta = param.delta
    N = param.N
    directory = param.path
    cnt = 0
    cnt_wrong = 0
    n_del = param.n_del
    while cnt < maxIter:
        X = htools.randX(param)
        D = htools.x2hdm(param, X)
        W = htools.randMask(param)
        output = HDM_metric(param,D, D, W,param.cost, param.norm)
        cvx_status = output.status
        if cvx_status == 'optimal':
            G = htools.h_rankPrj(output.G, param)
            G = htools.valid(G)
            error_out = htools.test_error(param, D, G, 1,np.ones((N,N)))
            cnt += 1
            if error_out > delta:
                cnt_wrong = cnt_wrong + 1
    prob = 1-cnt_wrong/maxIter
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(directory+'/n_dlt_'+'_N_'+str(N)+'_ndlt_'+str(n_del)+'.npy', n_del)
    np.save(directory+'/p_'+'_N_'+str(N)+'_ndlt_'+str(n_del)+'.npy', prob)
    return prob
###########################################################################
def HDM_order_sensitivity(param, D, index):
    N = param.N
    d = param.d
    #######################################
    output = HDM_OUT()
    EPS = 0.01
    #######################################
    Gp = cvx.Variable((N, N), PSD=True)
    Gn = cvx.Variable((N, N), PSD=True)
    G = Gp-Gn
    #######################################
    con = []
    con.append(cvx.diag(G) == -1)
    con.append(G <= -1)

    M = np.shape(index)[0]
    for m in range(M):
        i1 = int(index[m,0])
        i2 = int(index[m,1])
        i3 = int(index[m,2])
        i4 = int(index[m,3])
        if D[i1,i2] < D[i3,i4]:
            con.append(G[i1,i2] - G[i3,i4] >= EPS)
        if D[i1,i2] > D[i3,i4]:
            con.append(G[i3,i4]- G[i1,i2] >= EPS)
    #######################################
    min_dist = 1
    np.fill_diagonal(D, 1000)
    k,l = np.unravel_index( (-D).argmax(), D.shape)
    for i in range(N):
        for j in range(i+1,N):
            if i == k and j == l:
                con.append(G[i,j] == htools._cosh(min_dist))
                continue
            if j == k and i == l:
                con.append(G[i,j] == htools._cosh(min_dist))
                continue
            con.append(G[i,j] <= htools._cosh(min_dist))
    #######################################
    cost = cvx.trace(Gn)+cvx.trace(Gp)
    #######################################
    obj = cvx.Minimize(cost)
    prob = cvx.Problem(obj,con)
    try:
        prob.solve(solver=cvx.CVXOPT,verbose=False,normalize = False, #max_iter= 500, 
                kktsolver = 'robust')#, refinement = 2)
    except Exception as message:
        print(message)
    output.status = str(prob.status)
    if str(prob.status) == 'optimal':
        G = htools.h_rankPrj(G.value, param)
        output.G        = G
    else:
        print('fail')
    return output
###########################################################################
def sensitivity(param,K,M,R):
    directory = param.path
    N= param.N
    index = htools.ordinal_set(param)
    N_choose_2 = int(N*(N-1)/2)
    N_choose_2_2 = int( N_choose_2 * ( N_choose_2-1)/2 )
    S = 1-R*N_choose_2/N_choose_2_2
    print('Sparsity is ', S)
    Dt = np.zeros((M,K,N,N))
    for m in range(M):   
        X = htools.randX(param)
        D = htools.x2hdm(param, X)    
        for k in range(K):
            index_set = np.random.choice(N_choose_2_2, R*N_choose_2,replace=False)
            index_k = index[index_set,:]
            output = HDM_order_sensitivity(param, D, index_k)
            Dt[m,k,:,:] = htools._arccosh(output.G)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(directory+'/D'+'_N_'+str(N)+'_S_'+str(S)+'.npy', Dt)
