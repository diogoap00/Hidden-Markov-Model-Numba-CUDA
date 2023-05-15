#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:13:24 2023

@author: diogoap00

This a library to work with hidden markov models with discrete observation space. 
It includes simulation functions, likelihood functions, and 
expectation maximization related functions.

The ind module has functions for the estimations of parameters assuming independent
segments of data points.
Some functions are written in numba.
"""

import numpy as np
from numba import jit
from hmmdisc import arraychecks as ac
from hmmdisc import basic as hmmb
import time
import warnings


####### Check Functions #######


def check_finind_params(params):
    """
    Checks if params is a valid tuple of parameters for a hidden markov model 
    with independent segments of data.
    The tuple should have 8 elements such that:
    element 0: should be a positive integer, N, denoting the number of hidden states
    element 1: should be a positive integer, M, denoting number of possible values for the observations.
    element 2: p_0, the initial condition for the hidden Markov process, 1 dim array
    element 3: pi the transition matrix of the hidden Markov process, 2 dim array
    element 4: matrix B, (M x N) matrix, denoting the emission probabilities for the observations.
    element 5: D, a positive integer denoting the segments size
    element 6: L, a non-negative integer such that L+1 denotes the number of segments
    element 7: U, a (N x L) matrix such that each column gives the initial conditions for each segment.
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
    
    ac.pos_int(D,name="D")
    ac.non_neg(L,name="L")
    ac.trans_matrix(U,shape=(N,L),name="U")


####### Likelihood Functions #######


@jit(nopython=True)
def numba_like_finind(Y,params):
    """
    Use like_finind for this function with parameter checks.
    Calculates the likelihood for the data Y and parameters params.
    params should be a tuple of parameters as in check_finind_params.
    The likelihood is the one for the independent segments problem
    """
    
    p_0=params[2]
    pi=params[3]
    B=params[4]
    D=params[5]
    L=params[6]
    U=params[7]
    
    T=len(Y)
    
    like=0
    
    for l in range(L+1):
        for d in range(D):
            
            k=l*D+d
            
            if k<T:
                
                if d==0:
                    if k==0:
                        gamma=B[Y[k]]*p_0
                        gamma_norm=np.sum(gamma)
                        alpha=gamma/gamma_norm
                        like=like+np.log(gamma_norm)
                    else:
                        gamma=B[Y[k]]*U[:,l-1]
                        gamma_norm=np.sum(gamma)
                        alpha=gamma/gamma_norm
                        like=like+np.log(gamma_norm)
                
                else:
                    gamma=B[Y[k]]*np.dot(pi,alpha)
                    gamma_norm=np.sum(gamma)
                    alpha=gamma/gamma_norm
                    
                    like=like+np.log(gamma_norm)
    
    return like
    
    
def like_finind(Y,params,dochecks=True):
    """
    Calculates the likelihood for the data Y and parameters params.
    params should be a tuple of parameters as in check_finind_params.
    The likelihood is the one for the independent segments problem.
    Parameters must satisfy D*L < len(Y) <= D*(L+1) .
    """
    
    warnings.simplefilter('ignore')
    
    if dochecks:
        ac.vector_index(Y,name="Y")
        check_finind_params(params)
        
        D=params[5]
        L=params[6]
        T=len(Y)
        
        if not ( D*L < T and T <= D*(L+1) ):
            raise ValueError("Parameters must satisfy D*L < len(Y) <= D*(L+1)")
    
    like=numba_like_finind(Y,params)
    
    warnings.simplefilter('default')
    
    return like


####### alpha beta Calculation Functions #######


@jit(nopython=True)
def numba_finind_mu_nu(Y,params):
    """
    Use finind_mu_nu for this function with parameter checks.
    Calculates the processes mu and nu for a given dataset Y
    and parameters params (as in check_finind_params).
    This is for the model with indenpendent segments.
    Parameters must satisfy D*L < len(Y) <= D*(L+1).
    returns (mu,nu)
    """
    
    N=params[0]
    p_0=params[2]
    pi=params[3]
    B=params[4]
    D=params[5]
    L=params[6]
    U=params[7]
    
    T=len(Y)
    
    if T<D*(L+1):
        nan_size=D*(L+1)-T
        nan_vec=np.zeros(nan_size,dtype="int32")
        Y=np.append(Y,nan_vec)
    
    mu=np.empty((N,L+1,D))
    nu=np.empty((N,L+1,D))
    
    mu_last=np.zeros((N,L+1))
    
    gamma_0=B[Y[0]]*p_0
    mu_last[:,0]=gamma_0/np.sum(gamma_0)
    
    ind_now=np.arange(L+1)*D
    gamma_s=np.transpose(B[Y[ind_now[1:]]])*U
    mu_last[:,1:]=gamma_s/np.sum(gamma_s,axis=0)
    
    mu[:,:,0]=mu_last
    
    for k in range(1,D):
        
        ind_now=ind_now+1
        
        gamma=np.transpose(B[Y[ind_now]])*np.dot(pi,mu_last)
        new_mu=gamma/np.sum(gamma,axis=0)
        
        mu[:,:,k]=new_mu
        mu_last=new_mu
        
    nu_last=np.ones((N,L+1))/N
    nu[:,:,-1]=1./N
    
    for k in range(1,D):
        
        gbg=np.transpose(B[Y[ind_now]])*nu_last
        tmp=np.dot(np.transpose(pi),gbg)
        tmp=tmp/np.sum(tmp,axis=0)
        nu[:,:,(-1-k)]=tmp
        
        nu_last=tmp
        
        if ind_now[-1]==T:
            nu[:,-1,-k-1]=1./N
            nu_last[:,-1]=1./N
        
        ind_now=ind_now-1
        
    mu=np.reshape(mu,(N,(L+1)*D))
    nu=np.reshape(nu,(N,(L+1)*D))  
    
    return(mu[:,:T],nu[:,:T])
        
    
def finind_mu_nu(Y,params,dochecks=True):
    """
    Calculates the processes mu and nu for a given dataset Y
    and parameters params (as in check_finind_params).
    This is for the model with indenpendent segments.
    Parameters must satisfy D*L < len(Y) <= D*(L+1).
    returns (mu,nu)
    """
    
    warnings.simplefilter('ignore')
    
    if dochecks:
        ac.vector_index(Y,name="Y")
        check_finind_params(params)
        
        D=params[5]
        L=params[6]
        T=len(Y)
        
        if not ( D*L < T and T <= D*(L+1) ):
            raise ValueError("Parameters must satisfy D*L < len(Y) <= D*(L+1)")
    
    albet=numba_finind_mu_nu(Y,params)
    
    warnings.simplefilter('default')
    
    return albet


####### Maximization step Functions #######


def max_step(Y,params,albet,dochecks=True):
    """
    Does the maximization step of the EM algorithm.
    params should be as in check_finind_params.
    albet are the mu, nu, matrices as the return in finind_mu_nu.
    returns new parameters as in check_finind_params.
    """
    
    warnings.simplefilter('ignore')

    if dochecks:
        ac.vector_index(Y,name="Y")
        check_finind_params(params)
        
        N=params[0]
        T=len(Y)
        
        mu,nu=albet
        ac.trans_matrix(mu,shape=(N,T),name="mu")
        ac.trans_matrix(nu,shape=(N,T),name="nu")
    
    N=params[0]
    M=params[1]
    pi=params[3]
    B=params[4]
    D=params[5]
    L=params[6]
    
    alpha,beta=albet
    
    T=len(Y)
    
    gbg=np.transpose(B[Y])*beta
    cond_trans_norms=1/np.sum(gbg[:,1:]*np.dot(pi,alpha[:,:-1]),axis=0)
    cond_trans_norms[(np.arange(T-1)+1)%D==0]=0
    cond_trans_sum=pi*np.dot(gbg[:,1:]*cond_trans_norms,np.transpose(alpha[:,:-1]))
    new_pi=cond_trans_sum/np.sum(cond_trans_sum,axis=0)
    
    inprod=np.sum(alpha*beta,axis=0)
    cond_prob=alpha*beta/inprod
    
    Y_ind=np.zeros((M,T))
    Y_ind[(Y,np.arange(T))]=1
    
    new_B=np.dot(Y_ind,np.transpose(cond_prob))
    new_B=new_B/np.sum(new_B,axis=0)
    
    new_p_0=alpha[:,0]*beta[:,0]/np.sum(alpha[:,0]*beta[:,0])
    
    new_U=cond_prob[:,np.arange(1,L+1)*D]
    
    warnings.simplefilter('default')
    
    return(N,M,new_p_0,new_pi,new_B,D,L,new_U)


####### Iteration EM Functions #######


def finind_em_it(Y,params,dochecks=True,report_time=False):
    """
    Performs an iteration of the expectation maximization and returns the
    updated parameters. params should be as in check_finind_params.
    """
    
    warnings.simplefilter('ignore')
    
    if report_time:
        t=time.perf_counter()
        
    albet=finind_mu_nu(Y,params,dochecks=dochecks)
    
    if report_time:
        print("albet: ",time.perf_counter()-t)
        t=time.perf_counter()
        
    new_params=max_step(Y,params,albet,dochecks=dochecks)
    
    if report_time:
        print("max step: ",time.perf_counter()-t)
    
    warnings.simplefilter('default')
    
    return new_params


# ### Debugging

# # Real params

# N=3
# M=3

# p_0=np.zeros(N)
# p_0[0]=1.

# pi=np.ones((N,N))
# for i in range(N):
#     pi[i,i]=pi[i,i]+2
    
# pi=pi/np.sum(pi,axis=0)

# B=np.ones((M,N))
# for j in range(min(M,N)):
#     B[j,j]=B[j,j]+2

# B=B/np.sum(B,axis=0)


# params=(N,M,p_0,pi,B)
# hmmb.check_fin_params(params)

# print(" ")
# for el in params:
#     print(el)
# print(" ")

# T=200000
    
# Y=hmmb.sim_fin(T,params)

# B_real=B
# pi_real=pi


# # paralel

# # starting params

# N=3
# M=3

# p_0=np.ones(N)/N

# pi=np.ones((N,N))
# for i in range(N):
#     pi[i,i]=pi[i,i]+2
    
# pi=pi/np.sum(pi,axis=0)

# B=np.ones((M,N))
# for j in range(min(M,N)):
#     B[j,j]=B[j,j]+2

# B=B/np.sum(B,axis=0)

# D=50
# L=T//D-1

# U=np.ones((N,L))/N

# params=(N,M,p_0,pi,B,D,L,U)
# check_finind_params(params)

# like_last=0

# for it in range(20):
    
#     params=finind_em_it(Y,params)
    
#     if it>0:
#         like=like_finind(Y, params)
#         print(like-like_last)
#         like_last=like
#         # print(like)
#         # print(hmmb.like_fin(Y,params))
    
#     # print(" ")
#     # print(params[2])
#     # print(params[3])
#     # print(params[4])
#     # print(" ")

# print(" ")
# print(params[2])
# print(params[3])
# print(params[4])
# print(" ")

# print(pi_real)
# print(B_real)







