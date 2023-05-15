#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 13:56:35 2022

@author: diogoap00


This a library to work with hidden markov models in continuous observation space. 
It includes simulation functions, likelihood functions, and 
expectation maximization related functions.

The basic module includes these functions implemented for CPU computation.
Some functions are written in numba.
"""


import numpy as np
from numba import jit
from hmmcont import arraychecks as ac
import time
import warnings



####### Check Functions #######


def check_ar_params(params):
    """
    Checks if params is a valid tuple of parameters for a hidden markov AR model with
    continuous observation space.
    The tuple should have 6 elements such that:
    element 0: should be a positive integer, N, denoting the number of hidden states
    element 1: should be a positive integer, M, where M-1 denotes the number of lags in the AR
    element 2: p_0, the initial condtion for the hidden Markov process, 1 dim array
    element 3: pi the transition matrix of the hidden Markov process, 2 dim array
    element 4: matrix A, (N x M) matrix, denoting the parameters of the AR for each state
    element 5: sigma, the vector of the variance parameters
    """
    
    N=params[0]
    M=params[1]
    p_0=params[2]
    pi=params[3]
    A=params[4]
    sigma=params[5]
    
    ac.pos_int(N,name="N")
    ac.pos_int(M,name="M")
    ac.prob_vector(p_0,size=N,name="p_0")
    ac.trans_matrix(pi,shape=(N,N),name="pi")
    ac.matrix(A,shape=(N,M),name="A")
    ac.vector(sigma,size=N,name="sigma")
    

def check_arp_params(params):
    """
    Checks if params is a valid tuple of parameters for a hidden markov AR model with
    continuous observation space (for the parallelized algorithm).
    The tuple should have 10 elements such that:
    element 0: should be a positive integer, N, denoting the number of hidden states
    element 1: should be a positive integer, M, denoting the number of lags in the AR
    element 2: p_0, the initial condtion for the hidden Markov process, 1 dim array
    element 3: pi the transition matrix of the hidden Markov process, 2 dim array
    element 4: matrix A, (N x M) matrix, denoting the parameters of the AR for each state
    element 5: sigma, the vector of the variance parameters
    element 6: D, a positive integer denoting the segment size
    element 7: L, a non-negative integer such that L+1 denotes the number of segments
    element 8: U, a (N x L) matrix such that each column gives the initial conditions for mu for each segment.
    element 9: V, a (N x L) matrix giving the initial conditions for nu in each segment.
    """
    
    N=params[0]
    M=params[1]
    p_0=params[2]
    pi=params[3]
    A=params[4]
    sigma=params[5]
    
    ac.pos_int(N,name="N")
    ac.pos_int(M,name="M")
    ac.prob_vector(p_0,size=N,name="p_0")
    ac.trans_matrix(pi,shape=(N,N),name="pi")
    ac.matrix(A,shape=(N,M),name="A")
    ac.vector(sigma,size=N,name="sigma")
    
    D=params[6]
    L=params[7]
    U=params[8]
    V=params[9]
    
    ac.pos_int(D,name="D")
    ac.non_neg(L,name="L")
    ac.trans_matrix(U,shape=(N,L),name="U")
    ac.trans_matrix(V,shape=(N,L),name="V")
    
    
####### Simulation Functions #######
    

@jit(nopython=True)
def numba_sim_ar(y_0,T,params):
    """
    Use sim_ar for this function with parameter checks.
    Simulates a path of a AR hidden makov model.
    y_0 is the initial values of the path of size M-1, T is the final time of the path,
    and params is a tuple of parameters as in check_ar_params.
    returns a np.array of size T+M-1, where the first elements are y_0
    """
    
    N=params[0]
    M=params[1]
    p_0=params[2]
    pi=params[3]
    A=params[4]
    sigma=params[5]
    
    Y=np.empty(T+M-1)
    Y[:(M-1)]=y_0
    
    rand=np.random.rand()
    pr_sum=0
    for i in range(N):
        pr_sum=pr_sum+p_0[i]
        if rand < pr_sum:
            X=i
            break
    
    A_flip=A[:,::-1]
    
    for k in range(M-1,T+M-1):
        Y[k] = A[X,0]+np.dot(A_flip[X,:-1],Y[(k-M+1):k]) + sigma[X]*np.random.normal()
        
        rand=np.random.rand()
        pr_sum=0
        for i in range(N):
            pr_sum=pr_sum+pi[i,X]
            if rand < pr_sum:
                X=i
                break
    
    return Y

    
    
def sim_ar(y_0,T,params,dochecks=True):
    """
    Simulates a path of a AR hidden makov model.
    y_0 is the initial values of the path of size M-1, T is the final time of the path,
    and params[:6] is a tuple of parameters as in check_ar_params.
    returns a np.array of size T+M-1, where the first elements are y_0
    """
    
    warnings.simplefilter('ignore')
    
    params=params[:6]
    
    if dochecks:
        ac.pos_int(T,name="T")
        check_ar_params(params)
        M=params[1]
        ac.vector(y_0,M-1,name="y_0")
        
    Y=numba_sim_ar(y_0,T,params)
    
    warnings.simplefilter('default')
    
    return Y


####### Likelihood Functions #######


@jit(nopython=True)
def numba_like_ar(Y,params):
    """
    Use like_ar for this function with parameter checks.
    Calculates the likelihood for the data Y and parameters params as in
    check_ar_params.
    """
    
    M=params[1]
    p_0=params[2]
    pi=params[3]
    A=params[4]
    sigma=params[5]
    
    A_flip=A[:,::-1]
    
    T=len(Y)-M+1
    
    like=0
    
    Y_now=Y[M-1]
    Y_past=Y[:(M-1)]
    
    pred=A[:,0]+np.dot(A_flip[:,:-1],Y_past)
    pred_error=1/2*(1/sigma*(Y_now-pred))**2 
    sqpi=1/np.sqrt(2*np.pi)
    G=1/sigma*sqpi*np.exp(-pred_error )
    
    gamma=G*p_0
    gamma_norm=np.sum(gamma)
    alpha=gamma/gamma_norm
    
    like=like+np.log(gamma_norm)
    
    for k in range(1,T):
        
        Y_now=Y[M-1+k]
        Y_past=Y[k:(M-1+k)]
        
        pred=A[:,0]+np.dot(A_flip[:,:-1],Y_past)
        pred_error=1/2*(1/sigma*(Y_now-pred))**2
        G=1/sigma*sqpi*np.exp(-pred_error)
        
        gamma=G*np.dot(pi,alpha)
        gamma_norm=np.sum(gamma)
        alpha=gamma/gamma_norm
        
        like=like+np.log(gamma_norm)
    
    return like
    
    
def like_ar(Y,params,dochecks=True):
    """
    Calculates the likelihood for the data Y and parameters params[:6] as in
    check_ar_params.
    """
    
    warnings.simplefilter('ignore')
    
    params=params[:6]
    
    if dochecks:
        ac.vector(Y,name="Y")
        M=params[1]
        if len(Y)<M: 
            raise ValueError("len(Y) must be at least M")
        check_ar_params(params)
        
    like=numba_like_ar(Y,params)
    
    warnings.simplefilter('default')
    
    return like

    
####### alpha beta Calculation Functions #######
    

@jit(nopython=True)
def numba_ar_alpha_beta(Y,params):
    """
    Use ar_alpha_beta for this function with parameter checks.
    Calculates the processes alpha and beta for a given set of data Y
    and parameters params (as in check_ar_params).
    returns (alpha,beta,G) , here G in the Gamma matrix.
    """
    
    N=params[0]
    M=params[1]
    p_0=params[2]
    pi=params[3]
    A=params[4]
    sigma=params[5]
    
    A_flip=A[:,::-1]
    
    T=len(Y)-M+1
    
    eb=Y.itemsize
    Y_window=np.lib.stride_tricks.as_strided(Y,(T,M-1),(eb,eb))
    
    pred=A[:,0]+np.dot(Y_window,np.transpose(A_flip[:,:-1]))
    pred_dif=np.transpose(Y[(M-1):]-np.transpose(pred))
    pred_error=1/2*(1/sigma*pred_dif)**2
    sqpi=1/np.sqrt(2*np.pi)
    G=np.transpose(1/sigma*sqpi*np.exp(-pred_error ))
    
    alpha=np.empty((N,T))
    beta=np.empty((N,T))
    
    gamma=G[:,0]*p_0
    alpha_last=gamma/np.sum(gamma)
    alpha[:,0]=alpha_last
    
    for k in range(1,T):
        
        gamma=G[:,k]*np.dot(pi,alpha_last)
        alpha_last=gamma/np.sum(gamma)
        alpha[:,k]=alpha_last

    beta[:,(T-1)]=1./N
    beta_last=np.ones(N)/N
    
    for k in range(1,T):

        gbg=G[:,T-k]*beta_last
        tmp=np.dot(np.transpose(pi),gbg)
        beta_last=tmp/np.sum(tmp)
        beta[:,T-k-1]=beta_last
    
    
    return (alpha,beta,G)


def ar_alpha_beta(Y,params,dochecks=True):
    """
    Calculates the processes alpha and beta for a given set of data Y
    and parameters params (as in check_ar_params).
    returns (alpha,beta,G) , here G in the Gamma matrix.
    """
    
    warnings.simplefilter('ignore')
    
    if dochecks:
        ac.vector(Y,name="Y")
        M=params[1]
        if len(Y)<M: 
            raise ValueError("len(Y) must be at least M")
        check_ar_params(params)
        
    albet=numba_ar_alpha_beta(Y,params)
        
    warnings.simplefilter('default')
    
    return albet


@jit(nopython=True)
def numba_arp_mu_nu(Y,params):
    """
    Use arp_alpha_nu for this function with parameter checks.
    Calculates the processes mu and nu for a given set of data Y
    and parameters params (as in check_arp_params).
    Parameters must satisfy 1+D*L < len(Y)-M+1 <= 1+D*(L+1)
    returns (mu,nu,G) , here G in the Gamma matrix.
    """
    
    N=params[0]
    M=params[1]
    p_0=params[2]
    pi=params[3]
    A=params[4]
    sigma=params[5]
    D=params[6]
    L=params[7]
    U=params[8]
    V=params[9]
    
    A_flip=A[:,::-1]
    A_ind=A[:,0]
    
    T=len(Y)-M+1
    
    if T<1+D*(L+1):
        nan_size=1+D*(L+1)-T
        nan_vec=np.empty(nan_size)
        nan_vec[:]=np.nan
        Y=np.append(Y,nan_vec)
    
    T_ext=len(Y)-M+1
    
    eb=Y.itemsize
    Y_window=np.lib.stride_tricks.as_strided(Y,(T_ext,M-1),(eb,eb))
    
    pred=A_ind+np.dot(Y_window,np.transpose(A_flip[:,:-1]))
    pred_dif=np.transpose(Y[(M-1):]-np.transpose(pred))
    pred_error=1/2*(1/sigma*pred_dif)**2
    sqpi=1/np.sqrt(2*np.pi)
    G=np.transpose(1/sigma*sqpi*np.exp(-pred_error ))
    
    mu=np.empty((N,L+1,D))
    nu=np.empty((N,L+1,D))
    
    gamma_0=G[:,0]*p_0
    mu_0=gamma_0/np.sum(gamma_0)
    
    mu_last=np.zeros((N,L+1))
    mu_last[:,0]=mu_0
    mu_last[:,1:]=U
    
    ind_now=np.arange(L+1)*D+1
    
    for k in range(D):
        
        gamma=G[:,ind_now]*np.dot(pi,mu_last)
        mu_last=gamma/np.sum(gamma,axis=0)
        
        mu[:,:,k]=mu_last
        ind_now=ind_now+1
    
    nu_last=np.empty((N,L+1))
    nu_last[:,:-1]=V
    nu_last[:,-1]=np.nan
    
    for k in range(D):
        
        ind_now=ind_now-1
        
        if ind_now[-1]==T-1:
            nu_last[:,-1]=1./N
        
        gbg=G[:,ind_now]*nu_last
        tmp=np.dot(np.transpose(pi),gbg)
        nu_last=tmp/np.sum(tmp,axis=0)
        nu[:,:,(-1-k)]=nu_last
        
        nu_last=nu[:,:,(-1-k)]
    
    mu=np.reshape(mu,(N,(L+1)*D))
    nu=np.reshape(nu,(N,(L+1)*D))
    
    mu_start=np.empty((N,1))
    mu_start[:,0]=mu_0
    mu=np.append(mu_start,mu[:,:T-1],axis=1)
    
    nu_end=np.empty((N,1))
    nu_end[:,0]=1./N
    nu=np.append(nu[:,:T-1],nu_end,axis=1)
    
    return (mu,nu,G[:,:T])
        
    
def arp_mu_nu(Y,params,dochecks=True):
    """
    Calculates the processes mu and nu for a given set of data Y
    and parameters params (as in check_arp_params).
    Parameters must satisfy 1+D*L < len(Y)-M+1 <= 1+D*(L+1)
    returns (mu,nu,G) , here G in the Gamma matrix.
    """
    
    warnings.simplefilter('ignore')
    
    if dochecks:
        
        ac.vector(Y,name="Y")
        M=params[1]
        if len(Y)<M: 
            raise ValueError("len(Y) must be at least M")
        check_arp_params(params)
        
        D=params[6]
        L=params[7]
        T=len(Y)-M+1
        
        if not ( 1+D*L < T and T <= 1+D*(L+1) ):
            raise ValueError("Parameters must satisfy 1+D*L < len(Y)-M <= 1+D*(L+1)")
    
    albet=numba_arp_mu_nu(Y,params)
    
    warnings.simplefilter('default')
    
    return albet

    
####### Maximization step Functions #######


@jit(nopython=True)
def numba_max_step(Y,params,albet):
    """
    Use max_step for this function with parameter checks.
    Does the maximization step of the EM algorithm.
    params can be as in check_ar_params or check_arp_params (both work).
    albet should be as the return of the ar_alpha_beta or arp_mu_nu functions
    returns parameters as the check_ar_params (Does not return the initial
    conditions for the segments U and V).
    """
    
    N=params[0]
    M=params[1]
    pi=params[3]
    
    alpha,beta,G=albet
    
    T=len(Y)-M+1
    
    eb=Y.itemsize
    Y_window=np.lib.stride_tricks.as_strided(Y,(T,M-1),(eb,eb))
    
    gbg=G*beta
    
    cond_trans_norms=np.sum(gbg[:,1:]*np.dot(pi,alpha[:,:-1]),axis=0)
    cond_trans_sum=pi*np.dot(gbg[:,1:]/cond_trans_norms,np.transpose(alpha[:,:-1]))
    new_pi=cond_trans_sum/np.sum(cond_trans_sum,axis=0)
    
    new_A=np.zeros((N,M))
    new_sigma=np.zeros(N)
    
    inprod=np.sum(alpha*beta,axis=0)
    cond_prob=alpha*beta/inprod
    
    
    Y_window_ext=np.hstack((np.asarray(Y_window),np.ones((T,1))))
    Y_st=Y[(M-1):]
    
    for i in range(N):
        
        cond_vec=cond_prob[i]
        Y_cond=np.transpose(Y_window_ext)*cond_vec
        
        sysmat=np.dot(Y_cond,Y_window_ext)
        sysvec=np.dot(Y_cond,Y_st)
        
        new_A[i]=np.linalg.solve(sysmat,sysvec)[::-1]
    
    new_A_flip=new_A[:,::-1]
    
    pred=new_A[:,0]+np.dot(Y_window,np.transpose(new_A_flip[:,:-1]))
    pred_dif=Y_st-np.transpose(pred)
    
    error_sq=np.sum((pred_dif**2)*cond_prob,axis=1)
    new_sigma=np.sqrt(error_sq/np.sum(cond_prob,axis=1))

    new_p_0=alpha[:,0]*beta[:,0]/np.sum(alpha[:,0]*beta[:,0])
    
    return(N,M,new_p_0,new_pi,new_A,new_sigma)


def max_step(Y,params,albet,dochecks=True):
    """
    Does the maximization step of the EM algorithm.
    params can be as in check_ar_params or check_arp_params (both work).
    albet should be as the return of the ar_alpha_beta or arp_alpha_beta functions
    returns parameters as the check_ar_params (Does not return the initial
    conditions for the segments U and V).
    """
    
    warnings.simplefilter('ignore')
    
    if dochecks:
        
        ac.vector(Y,name="Y")
        M=params[1]
        if len(Y)<M: 
            raise ValueError("len(Y) must be at least M")
        check_ar_params(params)
        
        N=params[0]
        
        T=len(Y)-M+1
        alpha,beta,G=albet
        ac.trans_matrix(alpha,shape=(N,T),name="alpha")
        ac.trans_matrix(beta,shape=(N,T),name="beta")
        ac.matrix(G,shape=(N,T),name="G")
    
    new_params=numba_max_step(Y,params,albet)
    
    warnings.simplefilter('default')
    
    return new_params


####### Iteration EM Functions #######
    

def ar_em_it(Y,params,dochecks=True):
    """
    Performs and iteration of the expectation maximization and returns the
    updated parameters
    """
    
    warnings.simplefilter('ignore')
        
    albet=ar_alpha_beta(Y,params,dochecks=dochecks)
    new_params=max_step(Y,params,albet,dochecks=dochecks)
    
    warnings.simplefilter('default')
    
    return new_params
    

def arp_em_it(Y,params,dochecks=True):
    """
    Performs an iteration of the parallelized EM algorithm. Returns the updated
    parameters as in check_arp_params             
    """
    
    warnings.simplefilter('ignore')
    
    albet=arp_mu_nu(Y,params,dochecks=dochecks)
    new_params_ar=max_step(Y,params,albet,dochecks=dochecks)
    
    D=params[6]
    L=params[7]
    
    mu=albet[0]
    nu=albet[1]
    
    ind_updt=(np.arange(L)+1)*D
    
    new_U=mu[:,ind_updt]
    new_V=nu[:,ind_updt]
    
    new_params=tuple(list(new_params_ar)+[D,L,new_U,new_V])
    
    warnings.simplefilter('default')
    
    return new_params
   
