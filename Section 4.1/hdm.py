import cvxpy as cvx
import os, sys
import numpy as np
import htools
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
###########################################################################
class HDM_OUT:
    def __init__(self):
        self.eo         = None
        self.ei         = None
        self.G          = None
        self.status     = None
###########################################################################
class EDM_OUT:
    def __init__(self):
        self.eo         = None
        self.ei         = None
        self.G          = None
        self.status     = None
###########################################################################
def HDM(param, D, index):
    N = param.N
    d = param.d
    #######################################
    output = HDM_OUT()
    X, Dn, Dn, W, ei = htools.generateData(param)
    output.ei       = ei
    W_vec = np.diag(W.flatten())
    G_n = htools._cosh(Dn)
    EPS = 5
    #######################################
    error_list = np.linspace(-1, -1, num=1)
    for error_log in error_list: 
        error = 10**error_log
        #######################################
        Gp = cvx.Variable((N, N), PSD=True)
        Gn = cvx.Variable((N, N), PSD=True)
        G = Gp-Gn
        #######################################
        con = []
        con.append( cvx.norm( cvx.matmul(W_vec, cvx.vec(G-G_n) ) )**2 <= error*W.sum())
        con.append(cvx.diag(G) == -1)
        con.append(G <= -1)
        M = np.shape(index)[0]
        for m in range(M):
            ei1 = htools.e(index[m,0],N)
            ei2 = htools.e(index[m,1],N)
            ei12 = np.matmul(ei1,ei2.T)
            ei3 = htools.e(index[m,2],N)
            ei4 = htools.e(index[m,3],N)
            ei34 = np.matmul(ei3,ei4.T)
            con.append(cvx.trace ( cvx.matmul(G,ei12-ei34) ) >= EPS)
        #######################################
        cost = cvx.trace(Gn)+cvx.trace(Gp)
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
            G = htools.h_rankPrj(G.value, param)
            eo = htools.test_error(param, D, G, 1)
            output.G        = G
            output.eo       = eo
            break
        print('fail')
    return output
###########################################################################
def HDM_order(param, D, index):
    N = param.N
    d = param.d
    M = np.shape(index)[0]
    #######################################
    output = HDM_OUT()
    EPS = 0.25
    p = 0.005
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
    #con.append(G <= -1)
    
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
    #max_dist = 10
    min_dist = 1
    k,l,i,j = htools.min_max_dist_odor(-D)
    con.append(G[k,l] == htools._cosh(min_dist))
    for i in range(N):
    	for j in range(i+1,N):
    		if i == k and j == l:
    			continue
    		if i == l and j == k:
    			continue
    		con.append(G[i,j] <= htools._cosh(min_dist))
    #######################################
    cost = cvx.trace(Gn)+cvx.trace(Gp)+(0.1/EPS)*cvx.sum(x)
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
    	#np.save('G_strawberry_full.npy', G.value)
    	#G = htools.h_rankPrj(G.value, param)
        output.G        = G.value
        output.eo       = x.value
        #output.eo       = eo
        #eo = htools.test_error(param, D, G, 1)
    else:
        print('fail')
    return output
###########################################################################
def EDM_order(param, D, index):
    N = param.N
    d = param.d
    M = np.shape(index)[0]
    #######################################
    output = HDM_OUT()
    EPS = 0.25
    #######################################
    G = cvx.Variable((N, N), PSD=True)
    x = cvx.Variable(M)
    one = np.ones((N,1))
    p = 0.005
    #######################################
    con = []
    con.append(G*one == 0)
    con.append(x >= 0)
    con.append(cvx.sum(x) <= EPS*p*M)
    
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
    #max_dist = 10
    min_dist = 1
    k,l,i,j = htools.min_max_dist_odor(-D)
    con.append(-2*G[k,l]+G[k,k]+G[l,l] == min_dist)
    for i in range(N):
    	for j in range(i+1,N):
    		if i == k and j == l:
    			continue
    		if i == l and j == k:
    			continue
    		con.append(-2*G[i,j]+G[i,i]+G[j,j] >= min_dist)
    #######################################
    cost = cvx.trace(G)+(0.1/EPS)*cvx.sum(x)
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
    	#np.save('G_tomato_full_EDM.npy', G.value)
    	#G = htools.h_rankPrj(G.value, param)
        output.G        = G.value
        output.eo       = x.value
        #eo = htools.test_error(param, D, G, 1)
    else:
        print('fail')
    return output
###########################################################################
def HDM_X(param,D,X, index):
    N = param.N
    d = param.d
    #######################################
    output = HDM_OUT()
    X_, D_, Dn_, W, ei = htools.generateData(param)
    Dn = D
    W_vec = np.diag(W.flatten())
    G_n = htools._cosh(Dn)
    EPS = 5
    #######################################
    error_list = np.linspace(-1, -1, num=1)
    for error_log in error_list: 
        error = 10**error_log
        #######################################
        Gp = cvx.Variable((N, N), PSD=True)
        Gn = cvx.Variable((N, N), PSD=True)
        G = Gp-Gn
        #######################################
        con = []
        con.append( cvx.norm( cvx.matmul(W_vec, cvx.vec(G-G_n) ) )**2 <= error*W.sum())
        con.append(cvx.diag(G) == -1)
        con.append(G <= -1)
        M = np.shape(index)[0]
        for m in range(M):
            ei1 = htools.e(index[m,0],N)
            ei2 = htools.e(index[m,1],N)
            ei12 = np.matmul(ei1,ei2.T)
            ei3 = htools.e(index[m,2],N)
            ei4 = htools.e(index[m,3],N)
            ei34 = np.matmul(ei3,ei4.T)
            con.append(cvx.trace ( cvx.matmul(G,ei12-ei34) ) >= EPS)#0*abs(abs(l1-l2) - abs(l3-l4)) )
        #######################################
        cost = cvx.trace(Gn)+cvx.trace(Gp)
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
            G = htools.h_rankPrj(G.value, param)
            eo = htools.test_error(param, D, G, 1)
            output.G        = G
            output.eo       = eo
            break
        print('fail')
    return output
###########################################################################
def HDM_fashion_mnist(param, D, W, K):
    from numpy.linalg import inv
    N = param.N
    d = param.d
    #######################################
    output = HDM_OUT()
    G_n = htools._cosh(D)
    #######################################
    Wp = np.eye(N)
    Wn = np.eye(N)
    Gp0 = np.eye(N)
    Gn0 = np.eye(N)
    error_list = np.linspace(1, 1, num=1) # Error thershold for estimated gramian (see semidefinite constraints)
    for error_log in error_list: 
        error = 10**error_log
        for k in range(K):
            #delta = 10**(-k/10)
            print(k)
        #######################################
            Gp = cvx.Variable((N, N), PSD=True)
            Gn = cvx.Variable((N, N), PSD=True)
            #w,v = np.linalg.eig(Gp0)
            #print(w[0:20])
            #w,v = np.linalg.eig(Gn0)
            #print(w[0:10])
            #print(Gp0[0:10,0])
            Gp.value = Gp0 # warm start values
            Gn.value = Gn0 # warm start values
            G = Gp-Gn
            #######################################
            con = []
            con.append( cvx.norm(  W*(G-G_n), 'fro' )**2 <= error*W.sum() )
            con.append(cvx.diag(G) == -1)
            con.append(G <= -1)

            # Square root of inverse gramian in previous estimation
            w,v = np.linalg.eig(Wp)
            w=w.real
            w[w<0] = 0
            v = v.real
            Wp_2 = np.matmul(np.matmul(v,np.diag(np.sqrt(w))),v.T)
            w,v = np.linalg.eig(Wn)
            w=w.real
            w[w<0] = 0
            v = v.real
            Wn_2 = np.matmul(np.matmul(v,np.diag(np.sqrt(w))),v.T)

            #######################################

            cost = cvx.norm(Wn_2*Gn,'fro')**2+cvx.norm(Wp_2*Gp,'fro')**2
            #######################################
            obj = cvx.Minimize(cost)
            prob = cvx.Problem(obj,con)
            try:
                prob.solve(solver=cvx.CVXOPT,verbose=True,normalize = False, #max_iter= 500, 
                kktsolver = 'robust')#, refinement = 2) # warm start should be enabled
            except Exception as message:
                print(message)
            output.status = str(prob.status)
            if str(prob.status) == 'optimal':
                Gp0 = Gp.value
                Gn0 = Gn.value

                w,v = np.linalg.eig(Gn0)
                w=w.real
                w[w<0] = 0
                v = v.real
                Gn0 = np.matmul(np.matmul(v,np.diag(w)),v.T)
                w,v = np.linalg.eig(Gp0)
                w=w.real
                w[w<0] = 0
                v = v.real
                Gp0 = np.matmul(np.matmul(v,np.diag(w)),v.T)
                output.G        = G.value
                #G = htools.h_rankPrj(G.value, param)
                #eo = htools.test_error(param, D, G, 1)
                #break
                #output.eo       = eo
                #Wp = np.linalg.pinv(Gp0+delta*np.eye(N))
                #Wn = np.linalg.pinv(Gn0+delta*np.eye(N))
                Wp = np.linalg.pinv(Gp0)
                Wn = np.linalg.pinv(Gn0)
    return output
###########################################################################
def Save(param, HDM_OUT, mode):
    import os
    import datetime
    now = str(datetime.datetime.now())
    path = param.path+now+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    if mode[0] == '1':
        np.save(path+'param',param)
    if mode[1] == '1':
        np.save(path+'HDM_OUTput',HDM_OUTput)
    if mode[2] == '1':
        np.save(path+'trj_output',trj_output)
###########################################################################
def rotation_ambiguity(param):
    d = param.d
    X0 = htools.X(param)
    Rb, Ru,b = htools.randomRbRu(param)
    b = b / np.linalg.norm(b)
    
    Xb = np.matmul(Rb,X0)
    Xu = np.matmul(Ru,X0)
    Xbu = np.matmul(Rb,Xu)

    Y0 = htools.loid2poincare(X0)
    Yb = htools.loid2poincare(Xb)
    Yu = htools.loid2poincare(Xu)
    Ybu = htools.loid2poincare(Xbu)

    theta = np.linspace(start = 0, stop = 2*np.pi, num = 200)

    N = Y0.shape[1]
    colors = np.random.rand(N)
    area = 100 * np.ones(N)  # 0 to 15 point radii
    plt.figure(0)
    plt.scatter(Y0[0,:], Y0[1,:], s=area, c=colors, alpha=0.5)
    plt.plot(np.cos(theta), np.sin(theta))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig('../fig.eps')
    plt.close()
    plt.figure(1)
    plt.scatter(Yb[0,:], Yb[1,:], s=area, c=colors, alpha=0.5)
    plt.plot(np.cos(theta), np.sin(theta))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig('../figb.eps')
    plt.close()
    plt.figure(2)
    plt.scatter(Yu[0,:], Yu[1,:], s=area, c=colors, alpha=0.5)
    plt.plot(np.cos(theta), np.sin(theta))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig('../figu.eps')
    plt.close()
    plt.figure(3)
    plt.scatter(Ybu[0,:], Ybu[1,:], s=area, c=colors, alpha=0.5)
    plt.plot(np.cos(theta), np.sin(theta))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig('../figbu.eps')
    plt.close()
    plt.figure(4)
    plt.plot(np.cos(theta), np.sin(theta))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis('equal')
    plt.axis('off')
    plt.arrow(0, 0, b[0], b[1], head_width=0.125, head_length=0.125) 
    plt.savefig('../circ.eps')
    plt.close()
    #plt.show()
###########################################################################
def tree_uclid_and_hyperbole(param, degree):
    D, adjM = htools.randomTree(param,degree)
    Dn = D + htools.sym_noise(param) 
    #######################################
    N = param.N
    colors = np.random.rand(N)
    theta = np.linspace(start = 0, stop = 2*np.pi, num = 200)
    area = 4*sum(adjM)  
    #######################################
    output_h = tree_embedding_hyperbole(D, Dn, param)
    print(output_h.eo)
    print(output_h.ei)
    G = output_h.G
    X = htools.hgram2x(param, G)
    Y = htools.loid2poincare(X)
    #######################################
    fig, ax = plt.subplots()
    plt.scatter(Y[0,:], Y[1,:], s=area, c=colors, alpha=1)
    plt.plot(np.cos(theta), np.sin(theta))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    for i in range(N):
        ax.annotate(i, (Y[0,i], Y[1,i]),size=3)
        j_list = adjM[i,:]
        j_list = np.nonzero(j_list)[0]
        for j in j_list:
            plt.plot([Y[0,i],Y[0,j]],[Y[1,i],Y[1,j]],'r-',linewidth=0.3)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig('../tree_hyperbole.eps')
    #######################################
    output_e = tree_embedding_euclid(D, Dn, param)
    print(output_e.eo)
    print(output_e.ei)
    G = output_e.G
    X = htools.gram2x(G, param)
    #######################################
    fig, ax = plt.subplots()
    plt.scatter(X[0,:], X[1,:], s=area, c=colors, alpha=1)
    for i in range(N):
        ax.annotate(i, (X[0,i], X[1,i]),size=3)
        j_list = adjM[i,:]
        j_list = np.nonzero(j_list)[0]
        for j in j_list:
            plt.plot([X[0,i],X[0,j]],[X[1,i],X[1,j]],'r-',linewidth=0.3)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig('../tree_euclid.eps')

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
    #con.append(G <= -1)

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
def SGD_metric(param,D, W):
    N = param.N
    d = param.d
    H = np.eye(d+1)
    H[0,0] = -1
    #######################################
    output = HDM_OUT()
    X = htools.randX(param)
    ee = 10^3
    G = htools._cosh(D)
    for i in range(10000):
        Gn = htools.x2lgram(param, X)
        #Gn = htools.valid(Gn)
        #Dn = htools._arccosh(G)
        E = G-Gn
        ee_1 = np.linalg.norm(np.multiply(W,Gn-G),'fro') / np.linalg.norm(np.multiply(W,G),'fro') 
        if ee_1 < 0.001 and i > 50 & math.isnan(ee_1):
            ee = ee_1
            #print(ee_1)
            #print(i)
            break
        else:
            ee = ee_1
        Z = np.zeros((d+1,N))
        for n1 in range(N):
            for n2 in range(N):
                if n1 == n2:
                    continue
                x1 = X[:,n1]
                x2 = X[:,n2]
                x12 = np.matmul(np.matmul(x1.T,H),x2)
                #c1 = 1/ np.sqrt(np.linalg.norm(x12)**2-1)
                c2 = np.sqrt((np.linalg.norm(x2)**2+1)/(np.linalg.norm(x1)**2+1))
                #diff = c1*(c2*x1-x2)
                diff = (c2*x1-x2)
                Z[:,n1] = Z[:,n1]+ W[n1,n2]*E[n1,n2]*diff
        X = X - (0.008)*(1/N)*Z
        for n in range(N):
            X[0,n] = np.sqrt(1+np.linalg.norm(X[1:d+1,n])**2);

    output.status = 'optimal'
    output.G = htools.x2lgram(param, X)
    return output
    
###########################################################################
def FindMaxSprsSGD(param):
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
        output = SGD_metric(param,D, W)
        cvx_status = output.status
        if cvx_status == 'optimal':
            G = htools.h_rankPrj(output.G, param)
            G = htools.valid(G)
            error_out = htools.test_error(param, D, G, 1,np.ones((N,N)))
            cnt += 1
            if error_out > delta:
                cnt_wrong = cnt_wrong + 1
            #print(error_out)
    prob = 1-cnt_wrong/maxIter
    print(prob)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(directory+'/n_dlt_'+'_N_'+str(N)+'_ndlt_'+str(n_del)+'.npy', n_del)
    np.save(directory+'/p_'+'_N_'+str(N)+'_ndlt_'+str(n_del)+'.npy', prob)
    return prob