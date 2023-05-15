#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 22:23:39 2022

@author: diogoap00


This a library to work with hidden markov models with discrete observation space. 
It includes simulation functions, likelihood functions, and 
expectation maximization related functions.

The basic module includes these functions implemented for CPU computation.
Some functions are written in numba.
"""


import numpy as np
from numba import jit
from hmmdisc import arraychecks as ac
import time
import warnings


####### Check Functions #######


def check_fin_params(params):
    """
    Checks if params is a valid tuple of parameters for a hidden markov model.
    The tuple should have 5 elements such that:
    element 0: should be a positive integer, N, denoting the number of hidden states
    element 1: should be a positive integer, M, denoting number of possible values for the observations.
    element 2: p_0, the initial condition for the hidden Markov process, 1 dim array
    element 3: pi the transition matrix of the hidden Markov process, 2 dim array
    element 4: matrix B, (M x N) matrix, denoting the emission probabilities for the observations.
    """
    
    N=params[0]
    M=params[1]
    p_0=params[2]
    pi=params[3]
    B=params[4]
    
    ac.pos_int(N,name="N")
    ac.pos_int(M,name="M")
    ac.prob_vector(p_0,size=N,name="p_0")
    ac.trans_matrix(pi,shape=(N,N),name="pi")
    ac.matrix(B,shape=(M,N),name="B")
    
    
def check_finp_params(params):
    """
    Checks if params is a valid tuple of parameters for a hidden markov model 
    with parallelization.
    The tuple should have 9 elements such that:
    element 0: should be a positive integer, N, denoting the number of hidden states
    element 1: should be a positive integer, M, denoting number of possible values for the observations.
    element 2: p_0, the initial condition for the hidden Markov process, 1 dim array
    element 3: pi the transition matrix of the hidden Markov process, 2 dim array
    element 4: matrix B, (M x N) matrix, denoting the emission probabilities for the observations.
    element 5: D, a positive integer denoting the segments size
    element 6: L, a positive integer such that L+1 denotes the number of segments
    element 7: U, a (N x L) matrix such that each column gives the initial conditions for each segment.
    element 8: V, a (N x L) matrix giving the initial conditions for beta in each segment.
    """
    
    N=params[0]
    M=params[1]
    p_0=params[2]
    pi=params[3]
    B=params[4]
    
    ac.pos_int(N,name="N")
    ac.pos_int(M,name="M")
    ac.prob_vector(p_0,size=N,name="p_0")
    ac.trans_matrix(pi,shape=(N,N),name="pi")
    ac.matrix(B,shape=(M,N),name="B")
    
    D=params[5]
    L=params[6]
    U=params[7]
    V=params[8]
    
    ac.pos_int(D,name="D")
    ac.pos_int(L,name="L")
    ac.trans_matrix(U,shape=(N,L),name="U")
    ac.trans_matrix(V,shape=(N,L),name="V")
    

####### Simulation Functions #######


@jit(nopython=True)
def numba_sim_fin(T,params):
    """
    Use sim_fin for this function with parameter checks.
    Simulates a path of the hidden makov model.
    params should be a tuple of parameters as in check_fin_params.
    returns a np.array of size T (dtype int32).
    """
    
    N=params[0]
    M=params[1]
    p_0=params[2]
    pi=params[3]
    B=params[4]
    
    rand=np.random.rand()
    pr_sum=0
    for i in range(N):
        pr_sum=pr_sum+p_0[i]
        if rand < pr_sum:
            X=i
            break
    
    Y=np.empty(T,dtype="int32")
    for k in range(T):
        
        rand=np.random.rand()
        pr_sum=0
        for j in range(M):
            pr_sum=pr_sum+B[j,X]
            if rand < pr_sum:
                Y[k]=j
                break

        rand=np.random.rand()
        pr_sum=0
        for i in range(N):
            pr_sum=pr_sum+pi[i,X]
            if rand < pr_sum:
                X=i
                break
    
    return Y
    

def sim_fin(T,params,dochecks=True):
    """
    Simulates a path of the hidden makov model.
    params[:5] should be a tuple of parameters as in check_fin_params.
    returns a np.array of size T (dtype int32).
    """
    
    warnings.simplefilter('ignore')
    
    params_fin=params[:5]
    
    if dochecks:
        ac.pos_int(T,name="T")
        check_fin_params(params_fin)
        
    Y=numba_sim_fin(T,params_fin)
    
    warnings.simplefilter('default')
    
    return Y
    

####### Likelihood Functions #######


@jit(nopython=True)
def numba_like_fin(Y,params):
    """
    Use like_fin for this function with parameter checks.
    Calculates the likelihood for the data Y and parameters params.
    params should be a tuple of parameters as in check_fin_params.
    """
    
    p_0=params[2]
    pi=params[3]
    B=params[4]
    
    T=len(Y)
    
    like=0
    
    gamma=B[Y[0]]*p_0
    gamma_norm=np.sum(gamma)
    alpha=gamma/gamma_norm
    
    like=like+np.log(gamma_norm)
    
    for k in range(1,T):

        gamma=B[Y[k]]*np.dot(pi,alpha)
        gamma_norm=np.sum(gamma)
        alpha=gamma/gamma_norm
        
        like=like+np.log(gamma_norm)
    
    return like
    
    
def like_fin(Y,params,dochecks=True):
    """
    Calculates the likelihood for the data Y and parameters params.
    params[:5] should be a tuple of parameters as in check_fin_params.
    """
    
    warnings.simplefilter('ignore')
    
    params_fin=params[:5]
    
    if dochecks:
        ac.vector_index(Y,name="Y")
        check_fin_params(params_fin)
        
    like=numba_like_fin(Y,params_fin)
    
    warnings.simplefilter('default')
    
    return like


####### alpha beta Calculation Functions #######


@jit(nopython=True)
def numba_fin_alpha_beta(Y,params):
    """
    Use fin_alpha_beta for this function with parameter checks.
    Calculates the processes alpha and beta for a given dataset Y
    and parameters params as in check_fin_params.
    returns (alpha,beta)
    """
    
    N=params[0]
    p_0=params[2]
    pi=params[3]
    B=params[4]
    
    T=len(Y)
    
    alpha=np.empty((N,T))
    beta=np.empty((N,T))
    
    gamma=B[Y[0]]*p_0
    alpha[:,0]=gamma/np.sum(gamma)
    
    for k in range(1,T):
        
        gamma=B[Y[k]]*np.dot(pi,alpha[:,k-1])
        alpha[:,k]=gamma/np.sum(gamma)

    beta[:,(T-1)]=1./N
    
    for k in range(1,T):

        gbg=B[Y[T-k]]*beta[:,T-k]
        tmp=np.dot(np.transpose(pi),gbg)
        beta[:,T-k-1]=tmp/np.sum(tmp)
    
    return (alpha,beta)


def fin_alpha_beta(Y,params,dochecks=True):
    """
    Calculates the processes alpha and beta for a given dataset Y
    and parameters params (as in check_fin_params).
    returns (alpha,beta)
    """
    
    warnings.simplefilter('ignore')
    
    if dochecks:
        ac.vector_index(Y,name="Y")
        check_fin_params(params)
        
    albet=numba_fin_alpha_beta(Y,params)
    
    warnings.simplefilter('default')
    
    return albet


@jit(nopython=True)
def numba_finp_mu_nu(Y,params):
    """
    Use finp_mu_nu for this function with parameter checks.
    Calculates the processes mu and nu for a given dataset Y
    and parameters params (as in check_finp_params).
    Parameters must satisfy 1+D*L < len(Y) <= 1+D*(L+1).
    returns (mu,nu)
    """
    
    N=params[0]
    p_0=params[2]
    pi=params[3]
    B=params[4]
    D=params[5]
    L=params[6]
    U=params[7]
    V=params[8]

    T=len(Y)
    
    if T<1+D*(L+1):
        nan_size=1+D*(L+1)-T
        nan_vec=np.zeros(nan_size,dtype="int32")
        Y=np.append(Y,nan_vec)
    
    mu=np.empty((N,L+1,D))
    nu=np.empty((N,L+1,D))
    
    gamma_0=B[Y[0]]*p_0
    mu_0=gamma_0/np.sum(gamma_0)
    
    mu_last=np.zeros((N,L+1))
    mu_last[:,0]=mu_0
    mu_last[:,1:]=U
    
    ind_now=np.arange(L+1)*D+1
    
    for k in range(D):
        
        gamma=np.transpose(B[Y[ind_now]])*np.dot(pi,mu_last)
        mu[:,:,k]=gamma/np.sum(gamma,axis=0)
        
        mu_last=mu[:,:,k]
        ind_now=ind_now+1
    
    nu_last=np.empty((N,L+1))
    nu_last[:,:-1]=V
    nu_last[:,-1]=np.nan
    
    for k in range(D):
        
        ind_now=ind_now-1
        
        if ind_now[-1]==T-1:
            nu_last[:,-1]=1./N
        
        gbg=np.transpose(B[Y[ind_now]])*nu_last
        tmp=np.dot(np.transpose(pi),gbg)
        tmp=tmp/np.sum(tmp,axis=0)
        nu[:,:,(-1-k)]=tmp
        
        nu_last=tmp
        
    mu=np.reshape(mu,(N,(L+1)*D))
    nu=np.reshape(nu,(N,(L+1)*D))  
    
    mu_start=np.empty((N,1))
    mu_start[:,0]=mu_0
    mu=np.append(mu_start,mu[:,:T-1],axis=1)
    
    nu_end=np.empty((N,1))
    nu_end[:,0]=1./N
    nu=np.append(nu[:,:T-1],nu_end,axis=1)
    
    return (mu,nu)
        
    
def finp_mu_nu(Y,params,dochecks=True):
    """
    Calculates the processes mu and nu for a given dataset Y
    and parameters params (as in check_finp_params).
    Parameters must satisfy 1+D*L < len(Y) <= 1+D*(L+1).
    returns (mu,nu)
    """
    
    warnings.simplefilter('ignore')
    
    if dochecks:
        ac.vector_index(Y,name="Y")
        check_finp_params(params)
        
        D=params[5]
        L=params[6]
        T=len(Y)
        
        if not ( 1+D*L < T and T <= 1+D*(L+1) ):
            raise ValueError("Parameters must satisfy 1+D*L < len(Y) <= 1+D*(L+1)")
    
    like=numba_finp_mu_nu(Y,params)
    
    warnings.simplefilter('default')
    
    return like


####### Maximization step Functions #######


def max_step(Y,params,albet,dochecks=True):
    """
    Does the maximization step of the EM algorithm.
    params[:5] should be as in check_fin_params.
    albet are the alpha beta matrices as the return in fin_alpha_beta.
    returns new parameters as in check_fin_params.
    This function also works for params as in check_finp_params and 
    albet as the return in finp_mu_nu.
    """
    
    warnings.simplefilter('ignore')
    
    params=params[:5]
    
    if dochecks:
        ac.vector_index(Y,name="Y")
        check_fin_params(params)
        
        N=params[0]
        T=len(Y)
        
        alpha,beta=albet
        ac.trans_matrix(alpha,shape=(N,T),name="alpha")
        ac.trans_matrix(beta,shape=(N,T),name="beta")
    
    N=params[0]
    M=params[1]
    pi=params[3]
    B=params[4]
    
    alpha,beta=albet
    
    T=len(Y)
    
    gbg=np.transpose(B[Y])*beta
    cond_trans_norms=np.sum(gbg[:,1:]*np.dot(pi,alpha[:,:-1]),axis=0)
    cond_trans_sum=pi*np.dot(gbg[:,1:]/cond_trans_norms,np.transpose(alpha[:,:-1]))
    new_pi=cond_trans_sum/np.sum(cond_trans_sum,axis=0)
    
    inprod=np.sum(alpha*beta,axis=0)
    cond_prob=alpha*beta/inprod
    
    Y_ind=np.zeros((M,T))
    Y_ind[(Y,np.arange(T))]=1
    
    new_B=np.dot(Y_ind,np.transpose(cond_prob))
    new_B=new_B/np.sum(new_B,axis=0)
    
    new_p_0=alpha[:,0]*beta[:,0]/np.sum(alpha[:,0]*beta[:,0])
    
    warnings.simplefilter('default')
    
    return(N,M,new_p_0,new_pi,new_B)


####### Iteration EM Functions #######
    

def fin_em_it(Y,params,dochecks=True,report_time=False):
    """
    Performs an iteration of the expectation maximization and returns the
    updated parameters. params should be as in check_fin_params.
    """
    
    warnings.simplefilter('ignore')
    
    if report_time:
        t=time.perf_counter()
        
    albet=fin_alpha_beta(Y,params,dochecks=dochecks)
    
    if report_time:
        print("albet: ",time.perf_counter()-t)
        t=time.perf_counter()
        
    new_params=max_step(Y,params,albet,dochecks=dochecks)
    
    if report_time:
        print("max step: ",time.perf_counter()-t)
        
    warnings.simplefilter('default')
    
    return new_params


def finp_em_it(Y,params,dochecks=True,report_time=False):
    """
    Performs an iteration of the parallelized EM algorithm. Returns the updated
    parameters. params should be as in check_finp_params.        
    """
    
    warnings.simplefilter('ignore')
    
    if report_time:
        t=time.perf_counter()
    
    albet=finp_mu_nu(Y,params,dochecks=dochecks)
    
    if report_time:
        print("albet: ",time.perf_counter()-t)
        t=time.perf_counter()
        
    new_params_fin=max_step(Y, params, albet, dochecks=dochecks)
    
    if report_time:
        print("max step: ",time.perf_counter()-t)
    
    D=params[5]
    L=params[6]
    
    mu=albet[0]
    nu=albet[1]
    
    ind_updt=(np.arange(L)+1)*D
    
    new_U=mu[:,ind_updt]
    new_V=nu[:,ind_updt]
    
    new_params=tuple(list(new_params_fin)+[D,L,new_U,new_V])
    
    warnings.simplefilter('default')
    
    return new_params


# ### Debugging

# # Real params

# N=4
# M=6

# p_0=np.zeros(N)
# p_0[0]=1.

# pi=np.ones((N,N))
# for i in range(N):
#     pi[i,i]=pi[i,i]+10
    
# pi=pi/np.sum(pi,axis=0)

# B=np.ones((M,N))
# for j in range(min(M,N)):
#     B[j,j]=B[j,j]+10

# B=B/np.sum(B,axis=0)


# params=(N,M,p_0,pi,B)
# check_fin_params(params)

# print(" ")
# for el in params:
#     print(el)
# print(" ")

# T=20000
    
# Y=sim_fin(T,params)

# B_real=B
# pi_real=pi



# ########

# # Baum-Welch

# # starting params

# N=4
# M=6

# p_0=np.ones(N)/N

# pi=np.ones((N,N))
# for i in range(N):
#     pi[i,i]=pi[i,i]+1
    
# pi=pi/np.sum(pi,axis=0)

# B=np.ones((M,N))
# for j in range(min(M,N)):
#     B[j,j]=B[j,j]+1

# B=B/np.sum(B,axis=0)


# params=(N,M,p_0,pi,B)
# check_fin_params(params)

# for it in range(10):
#     params=fin_em_it(Y,params)
#     print(like_fin(Y, params))

#     # print(" ")
#     # for el in params:
#     #     print(el)
#     # print(" ")

# print(pi_real)
# print(B_real)



# # paralel

# # starting params

# N=4
# M=6

# p_0=np.ones(N)/N

# pi=np.ones((N,N))
# for i in range(N):
#     pi[i,i]=pi[i,i]+1
    
# pi=pi/np.sum(pi,axis=0)

# B=np.ones((M,N))
# for j in range(min(M,N)):
#     B[j,j]=B[j,j]+1

# B=B/np.sum(B,axis=0)

# D=50
# L=T//D-1

# U=np.ones((N,L))/N
# V=np.ones((N,L))/N


# params=(N,M,p_0,pi,B,D,L,U,V)
# check_finp_params(params)


# for it in range(10):
#     params=finp_em_it(Y,params)
#     print(like_fin(Y, params))
    
#     print(" ")
#     print(params[2])
#     print(params[3])
#     print(params[4])
#     print(" ")

# print(" ")
# print(params[2])
# print(params[3])
# print(params[4])
# print(" ")

# print(pi_real)
# print(B_real)


