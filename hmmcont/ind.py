#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:07:05 2023

@author: diogoap00


This a library to work with hidden markov models in continuous observation space. 
It includes simulation functions, likelihood functions, and 
expectation maximization related functions.

The ind module has functions for the estimations of parameters assuming independent
segments of data points.
Some functions are written in numba.
"""

import numpy as np
from numba import jit
from hmmcont import arraychecks as ac
from hmmcont import basic as hmmb
import time
import warnings


####### Check Functions #######


def check_arind_params(params):
    """
    Checks if params is a valid tuple of parameters for a hidden markov AR model with 
    independent segments.
    The tuple should have 9 elements such that:
    element 0: should be a positive integer, N, denoting the number of hidden states
    element 1: should be a positive integer, M, here M-1 denotes the number of lags in the AR
    element 2: p_0, the initial condtion for the hidden Markov process, 1 dim array
    element 3: pi the transition matrix of the hidden Markov process, 2 dim array
    element 4: matrix A, (N x M) matrix, denoting the parameters of the AR for each state
    element 5: sigma, the vector of the variance parameters
    element 6: D, a positive integer denoting the segments size
    element 7: L, a non-negative integer such that L+1 denotes the number of segments
    element 8: U, a (N x L) matrix such that each column gives the initial conditions for each segment.
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
    
    ac.pos_int(D,name="D")
    ac.non_neg(L,name="L")
    ac.trans_matrix(U,shape=(N,L),name="U")
    
    
####### mu nu Calculation Functions #######


@jit(nopython=True)
def numba_like_ar(Y,params):
    """
    Use like_arind for this function with parameter checks.
    Calculates the likelihood for the data Y and parameters params as in
    check_arind_params.
    """
    
    M=params[1]
    p_0=params[2]
    pi=params[3]
    A=params[4]
    sigma=params[5]
    
    D=params[6]
    L=params[7]
    U=params[8]
    
    A_flip=A[:,::-1]
    
    T=len(Y)-M+1
    
    sqpi=1/np.sqrt(2*np.pi)
    like=0
    
    for l in range(L+1):
        for d in range(D):
            
            k=l*D+d
            
            if k<T:
                
                Y_now=Y[M-1+k]
                Y_past=Y[k:(M-1+k)]
                
                pred=A[:,0]+np.dot(A_flip[:,:-1],Y_past)
                pred_error=1/2*(1/sigma*(Y_now-pred))**2
                G=1/sigma*sqpi*np.exp(-pred_error)
                
                if d==0:
                    if k==0:
                        gamma=G*p_0
                        gamma_norm=np.sum(gamma)
                        alpha=gamma/gamma_norm
                        like=like+np.log(gamma_norm)
                    else:
                        gamma=G*U[:,l-1]
                        gamma_norm=np.sum(gamma)
                        alpha=gamma/gamma_norm
                        like=like+np.log(gamma_norm)
                
                else:
                    gamma=G*np.dot(pi,alpha)
                    gamma_norm=np.sum(gamma)
                    alpha=gamma/gamma_norm
                    like=like+np.log(gamma_norm)
    
    return like
    
    
def like_arind(Y,params,dochecks=True):
    """
    Calculates the likelihood for the data Y and parameters params as in
    check_arind_params.
    """
    
    warnings.simplefilter('ignore')
    
    if dochecks:
        ac.vector(Y,name="Y")
        M=params[1]
        if len(Y)<M: 
            raise ValueError("len(Y) must be at least M")
        check_arind_params(params)
        
    like=numba_like_ar(Y,params)
    
    warnings.simplefilter('default')
    
    return like

    
####### mu nu Calculation Functions #######
    

@jit(nopython=True)
def numba_arind_mu_nu(Y,params):
    """
    Use arind_mu_nu for this function with parameter checks.
    Calculates the processes mu and nu for a given set of data Y
    and parameters params (as in check_arind_params).
    Parameters must satisfy D*L < len(Y)-M+1 <= D*(L+1)
    returns mu,nu,G
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
    
    A_flip=A[:,::-1]
    A_ind=A[:,0]
    
    T=len(Y)-M+1
    
    if T<D*(L+1):
        nan_size=D*(L+1)-T
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
    mu[:,0,0]=gamma_0/np.sum(gamma_0)
    
    ind_now=np.arange(L+1)*D
    
    gamma=G[:,ind_now[1:]]*U
    mu[:,1:,0]=gamma/np.sum(gamma,axis=0)
    
    mu_last=mu[:,:,0]
    
    for k in range(1,D):
        
        ind_now=ind_now+1
        
        gamma=G[:,ind_now]*np.dot(pi,mu_last)
        mu_last=gamma/np.sum(gamma,axis=0)
        
        mu[:,:,k]=mu_last
    
    nu[:,:,-1]=1./N
    nu_last=np.ones((N,L+1))/N
    
    for k in range(1,D):

        gbg=G[:,ind_now]*nu_last
        tmp=np.dot(np.transpose(pi),gbg)
        nu_last=tmp/np.sum(tmp,axis=0)

        if ind_now[-1]==T:
            nu_last[:,L,]=1./N
        
        nu[:,:,(-1-k)]=nu_last
        
        ind_now=ind_now-1
    
    mu=np.reshape(mu,(N,(L+1)*D))
    nu=np.reshape(nu,(N,(L+1)*D))
    
    mu=mu[:,:T]
    nu=nu[:,:T]

    return (mu,nu,G[:,:T])
        
    
def arind_mu_nu(Y,params,dochecks=True):
    """
    Calculates the processes mu and nu for a given set of data Y
    and parameters params (as in check_arind_params).
    Parameters must satisfy D*L < len(Y)-M+1 <= D*(L+1)
    returns mu,nu,G
    """
    
    warnings.simplefilter('ignore')
    
    if dochecks:
        
        ac.vector(Y,name="Y")
        M=params[1]
        if len(Y)<M: 
            raise ValueError("len(Y) must be at least M")
        check_arind_params(params)
        
        D=params[6]
        L=params[7]
        T=len(Y)-M+1
        
        if not ( D*L < T and T <= D*(L+1) ):
            raise ValueError("Parameters must satisfy D*L < len(Y)-M+1 <= D*(L+1)")
    
    albet=numba_arind_mu_nu(Y,params)
    
    warnings.simplefilter('default')
    
    return albet


####### Maximization step Functions #######


@jit(nopython=True)
def numba_max_step(Y,params,albet):
    """
    Use max_step for this function with parameter checks.
    Does the maximization step of the EM algorithm.
    params should be as in check_arind_params.
    returns parameters as in check_ar_params, i.e., the first 6 elements of the tuple.
    """
    
    N=params[0]
    M=params[1]
    pi=params[3]
    D=params[6]
    
    mu,nu,G=albet
    
    T=len(Y)-M+1
    
    eb=Y.itemsize
    Y_window=np.lib.stride_tricks.as_strided(Y,(T,M-1),(eb,eb))
    
    gbg=G*nu
    
    cond_trans_norms=1/np.sum(gbg[:,1:]*np.dot(pi,mu[:,:-1]),axis=0)
    cond_trans_norms[(np.arange(T-1)+1)%D==0]=0
    
    cond_trans_sum=pi*np.dot(gbg[:,1:]*cond_trans_norms,np.transpose(mu[:,:-1]))
    new_pi=cond_trans_sum/np.sum(cond_trans_sum,axis=0)
    
    new_A=np.zeros((N,M))
    new_sigma=np.zeros(N)
    
    inprod=np.sum(mu*nu,axis=0)
    cond_prob=mu*nu/inprod
    
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

    new_p_0=mu[:,0]*nu[:,0]/np.sum(mu[:,0]*nu[:,0])
    
    return(N,M,new_p_0,new_pi,new_A,new_sigma)


def max_step(Y,params,albet,dochecks=True):
    """
    Does the maximization step of the EM algorithm.
    params should be as in check_arind_params.
    returns parameters as in check_ar_params, i.e., the first 6 elements of the tuple.
    """
    
    warnings.simplefilter('ignore')
    
    if dochecks:
        
        ac.vector(Y,name="Y")
        M=params[1]
        if len(Y)<M: 
            raise ValueError("len(Y) must be at least M")
        check_arind_params(params)
        
        N=params[0]
        
        T=len(Y)-M+1
        mu,nu,G=albet
        ac.trans_matrix(mu,shape=(N,T),name="mu")
        ac.trans_matrix(nu,shape=(N,T),name="nu")
        ac.matrix(G,shape=(N,T),name="G")
        
    new_params=numba_max_step(Y,params,albet)
        
    warnings.simplefilter('default')
    
    return new_params


####### Iteration EM Functions #######


def arind_em_it(Y,params,dochecks=True):
    """
    Performs an iteration of the ind EM algorithm. Returns the updated
    parameters as in check_arind_params             
    """
    
    warnings.simplefilter('ignore')
    
    albet=arind_mu_nu(Y,params,dochecks=dochecks)

    new_params_ar=max_step(Y,params,albet,dochecks=dochecks)
    
    D=params[6]
    L=params[7]
    
    mu,nu,G=albet
    
    ind_updt=(np.arange(L)+1)*D
    
    new_U=mu[:,ind_updt]*nu[:,ind_updt]
    new_U=new_U/np.sum(new_U,axis=0)
    
    new_params=tuple(list(new_params_ar)+[D,L,new_U])
    
    warnings.simplefilter('default')
    
    return new_params

