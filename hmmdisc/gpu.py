# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 15:10:30 2023

@author: Diogo Pereira

This a library to work with hidden markov models with discrete observation space. 
It includes simulation functions, likelihood functions, and 
expectation maximization related functions.

The gpu module implements the basic.finp_em_it in a cuda implementation.
"""


import numpy as np
from numba import cuda, float64
from hmmdisc import basic as hmmb
from hmmdisc import arraychecks as ac
import time
import math
import warnings




def finp_cuda_em_it(Y,params,iters=1,block_size=32,n_blocks=128,report_time=False):
    """
    Equivalent function to basic.finp_em_it but uses cuda (Nvidia GPU).
    Y should be a np.array of data and params as basic.check_finp_params.
    
    Use iters for the number of iterations to do. It is recommended to use this
    if one is doing more then one iteration because moving GPU memory to CPU memory is slow.
    
    block_size is the size of blocks in the computation of mu and nu. It is required
    that N <= block_size . Also there are limitations of the GPU for the block_size maximum.
    Also because of GPU specifications, block_size is always a multiple of 32 (if this is
    not the case the next biggest multiple is chosen automaticly)
    
    The computation is divided in sets of N operations (each segment), as such 
    it is recommended to to choose block_size such that (block_size % N) is small.
    The remainder will result in this ammount of idle threads.
    
    n_blocks is used in the sum reductions in the maximization step. In general it
    is only required that n_blocks < T (It is recommended to use a number much smaller). 
    """
    
    warnings.simplefilter('ignore')
    
    ##### Parameter Checks #####
    
    ac.vector_index(Y,name="Y")
    hmmb.check_finp_params(params)
    
    N=params[0]
    M=params[1]
    p_0=params[2]
    pi=params[3]
    B=params[4]
    D=params[5]
    L=params[6]
    U=params[7]
    V=params[8]
    
    T=len(Y)
    if not ( 1+D*L < T and T <= 1+D*(L+1) ):
        raise ValueError("Parameters must satisfy 1+D*L < len(Y) <= 1+D*(L+1)")
    
    if block_size < N :
        raise ValueError("block_size must be greater or equal to N (default is 32)")
        
        
    ##### Functions #####
    
    # constants to be used
    
    SEGBLOCKS=int(block_size//N)
    MUBLOCKS=int(np.ceil((L+1)/SEGBLOCKS))
    
    NSUMSSIZE=2**math.ceil(math.log(N,2)-1)
    if N==1: NSUMSSIZE=1
    
    PIINITBLOCKS=int(np.ceil(N**2/(32)))
    BINITBLOCKS=int(np.ceil(N*M/(32)))
    
    NORMBLOCKS=int(np.ceil(N/(32)))
    UVBLOCKS=int(np.ceil(N*L/(32)))
    
    
    @cuda.jit
    def mu_nu_cuda(Y,p_0,pi,B,U,V,mu,nu):
        """
        Calculates the mu nu matrices, given some parameters and dataset Y.
        The mu and nu matrices are done in parallel.
        
        Designed to use:
        threads per block : block_size
        blocks per grid   : 2*MUBLOCKS
        """        
        
        thread = cuda.threadIdx.x
        seg = thread // N
        i = thread % N
        
        b=cuda.blockIdx.x
        
        tmp_last=cuda.shared.array((N,SEGBLOCKS), dtype=float64)
        tmp_sum=cuda.shared.array((NSUMSSIZE,SEGBLOCKS), dtype=float64)
        tmp=cuda.shared.array((N,SEGBLOCKS), dtype=float64)
        gbg_tmp=cuda.shared.array((N,SEGBLOCKS), dtype=float64)
        
        if seg<SEGBLOCKS:
            
            if b<MUBLOCKS:
                
                l=b*SEGBLOCKS+seg
            
                if l==0:
        
                    tmp_last[i,0]=B[Y[0],i]*p_0[i]
                    cuda.syncthreads()
                    
                    if i<N-NSUMSSIZE:
                        tmp_sum[i,0]=tmp_last[i,0]+tmp_last[i+NSUMSSIZE,0]
                    else: 
                        if i<NSUMSSIZE:
                            tmp_sum[i,0]=tmp_last[i,0]
                    cuda.syncthreads()
                    
                    currsze=NSUMSSIZE//2
                    while(currsze>0):
                        if i<currsze:
                            tmp_sum[i,0] += tmp_sum[i+currsze,0]
                        cuda.syncthreads()
                        currsze//=2
                    
                    tmp_last[i,0]=tmp_last[i,0]/tmp_sum[0,0]
                    mu[i,0]=tmp_last[i,0]
                
                if l>0 and l<L+1:
                    tmp_last[i,seg]=U[i,l-1]
        
                cuda.syncthreads()
            
                for k in range(0,D):
                    
                    index=1+k+l*D
                    
                    if index<T:
                        
                        tmp[i,seg]=pi[i,0]*tmp_last[0,seg]
                        for j in range(1,N):
                            tmp[i,seg]+=pi[i,j]*tmp_last[j,seg]
                            
                        tmp[i,seg]=B[Y[index],i]*tmp[i,seg]
                        cuda.syncthreads()
                        
                        if i<N-NSUMSSIZE:
                            tmp_sum[i,seg]=tmp[i,seg]+tmp[i+NSUMSSIZE,seg]
                        else: 
                            if i<NSUMSSIZE:
                                tmp_sum[i,seg]=tmp[i,seg]
                        cuda.syncthreads()
                        
                        currsze=NSUMSSIZE//2
                        while(currsze>0):
                            if i<currsze:
                                tmp_sum[i,seg] += tmp_sum[i+currsze,seg]
                            cuda.syncthreads()
                            currsze//=2
        
                        
                        tmp_last[i,seg]=tmp[i,seg]/tmp_sum[0,seg]
                        mu[i,index]=tmp_last[i,seg]
                        cuda.syncthreads()
                        
            if b>=MUBLOCKS and b<2*MUBLOCKS:
                
                l=(b-MUBLOCKS)*SEGBLOCKS+seg
                
                if l<L:
                    tmp_last[i,seg]=V[i,l]
                
                cuda.syncthreads()
                
                for k in range(0,D):
                    
                    index=(l+1)*D-k
                    
                    if index==T-1:
                        
                        tmp_last[i,seg]=1./N
                        nu[i,index]=1./N
                    
                    if index<T:
                        
                        gbg_tmp[i,seg]=B[Y[index],i]*tmp_last[i,seg]

                        cuda.syncthreads()
                        
                        tmp[i,seg]=pi[0,i]*gbg_tmp[0,seg]
                        for j in range(1,N):
                            tmp[i,seg]+=pi[j,i]*gbg_tmp[j,seg]
                        cuda.syncthreads()
                        
                        if i<N-NSUMSSIZE:
                            tmp_sum[i,seg]=tmp[i,seg]+tmp[i+NSUMSSIZE,seg]
                        else: 
                            if i<NSUMSSIZE:
                                tmp_sum[i,seg]=tmp[i,seg]
                        cuda.syncthreads()
                        
                        currsze=NSUMSSIZE//2
                        while(currsze>0):
                            if i<currsze:
                                tmp_sum[i,seg] += tmp_sum[i+currsze,seg]
                            cuda.syncthreads()
                            currsze//=2
        
                        
                        tmp_last[i,seg]=tmp[i,seg]/tmp_sum[0,seg]
                        nu[i,index-1]=tmp_last[i,seg]
                        cuda.syncthreads()
                        
    
    @cuda.jit
    def pi_zero_init(pi):
        """
        Function to initialize a (N x N) matrix to zero
        
        Designed to use:
        threads per block : 32
        blocks per grid   : PIINITBLOCKS
        """
        
        ind=cuda.grid(1)
        
        if ind < N*N :
            
            i=ind//N
            j=ind%N
            
            pi[i,j]=0
    
    
    @cuda.jit
    def B_zero_init(B):
        """
        Function to initialize a (M x N) matrix to zero
        
        Designed to use:
        threads per block : 32
        blocks per grid   : BINITBLOCKS
        """
        
        ind=cuda.grid(1)
        
        if ind < M*N:
            
            m=ind//N
            i=ind%N
            
            B[m,i]=0
    
    
    @cuda.jit
    def pi_calc(Y,pi,B,mu,nu,new_pi):
        """
        Calculates the new matrix pi, given mu and nu. Matrix new_pi will not be
        such that each column sums to 1, requires an additional function for this.
        
        Designed to use:
        threads per block : block_size
        blocks per grid   : n_blocks
        """
        
        thread = cuda.threadIdx.x
        seg = thread // N
        i = thread % N
        
        b=cuda.blockIdx.x
        
        gbg_tmp=cuda.shared.array((N,SEGBLOCKS), dtype=float64)
        mat_tmp=cuda.shared.array((N,N,SEGBLOCKS), dtype=float64)
        mat_tmp_sum=cuda.shared.array((N,SEGBLOCKS), dtype=float64)
        
        pi_tmp=cuda.shared.array((N,N,SEGBLOCKS), dtype=float64)
        
        if seg<SEGBLOCKS:
            
            for j in range(0,N):
                pi_tmp[i,j,seg]=0
            
            for k in range(b*SEGBLOCKS+seg, T-1, n_blocks*SEGBLOCKS):
                
                Y_k=Y[k+1]
                mat_tmp_sum[i,seg]=0
                cuda.syncthreads()
                
                gbg_tmp[i,seg]=nu[i,k+1]*B[Y_k,i]
                for j in range(0,N):
                    mat_tmp[i,j,seg]=gbg_tmp[i,seg]*pi[i,j]*mu[j,k]
                    mat_tmp_sum[i,seg]+=mat_tmp[i,j,seg]
                cuda.syncthreads()
                
                if i<N-NSUMSSIZE:
                    mat_tmp_sum[i,seg]+=mat_tmp_sum[i+NSUMSSIZE,seg]
                cuda.syncthreads()
                
                currsze=NSUMSSIZE//2
                while(currsze>0):
                    if i<currsze:
                        mat_tmp_sum[i,seg] += mat_tmp_sum[i+currsze,seg]
                    cuda.syncthreads()
                    currsze//=2
                
                for j in range(0,N):
                    pi_tmp[i,j,seg]+=mat_tmp[i,j,seg]/mat_tmp_sum[0,seg]
            
            for j in range(0,N):
                cuda.atomic.add(new_pi, (i,j), pi_tmp[i,j,seg])
                
    
    @cuda.jit
    def B_calc(Y,mu,nu,new_B,new_p_0):
        """
        Calculates the new matrix B and p_0 (given mu and nu). Matrix new_B will not be
        such that each column sums to 1, requires an additional function for this.
        
        Designed to use:
        threads per block : block_size
        blocks per grid   : n_blocks
        """
        
        thread = cuda.threadIdx.x
        seg = thread // N
        i = thread % N
        
        b=cuda.blockIdx.x
        
        cond_vec=cuda.shared.array((N,SEGBLOCKS), dtype=float64)
        cond_vec_sum=cuda.shared.array((NSUMSSIZE,SEGBLOCKS), dtype=float64)
        B_tmp=cuda.shared.array((M,N,SEGBLOCKS), dtype=float64)
        
        if seg<SEGBLOCKS:
            
            for m in range(0,M):
                B_tmp[m,i,seg]=0
            
            for k in range(b*SEGBLOCKS+seg, T, n_blocks*SEGBLOCKS):
                
                cond_vec[i,seg]=mu[i,k]*nu[i,k]
                cuda.syncthreads()
                
                if i<N-NSUMSSIZE:
                    cond_vec_sum[i,seg]=cond_vec[i,seg]+cond_vec[i+NSUMSSIZE,seg]
                else: 
                    if i<NSUMSSIZE:
                        cond_vec_sum[i,seg]=cond_vec[i,seg]
                cuda.syncthreads()
                
                currsze=NSUMSSIZE//2
                while(currsze>0):
                    if i<currsze:
                        cond_vec_sum[i,seg] += cond_vec_sum[i+currsze,seg]
                    cuda.syncthreads()
                    currsze//=2
                
                B_tmp[Y[k],i,seg]+=cond_vec[i,seg]/cond_vec_sum[0,seg]
                
                if k==0:
                    new_p_0[i]=cond_vec[i,seg]/cond_vec_sum[0,seg]
                
            for m in range(0,M):
                cuda.atomic.add(new_B, (m,i), B_tmp[m,i,seg])
    
    
    @cuda.jit
    def B_normalize(new_B):
        """
        Normalizes a (M x N) matrix by dividing each element by the sum of the
        respective column. To be used after B_calc.
        
        Designed to use:
        threads per block : 32
        blocks per grid   : NORMBLOCKS
        """
        
        i = cuda.grid(1)
        
        if i < N :

            s=0
            for m in range(0,M):
                s+=new_B[m,i]
            
            for m in range(0,M):
                new_B[m,i]/=s
                
    
    @cuda.jit
    def pi_normalize(new_pi):
        """
        Normalizes a (N x N) matrix by dividing each element by the sum of the
        respective column. To be used after pi_calc.
        
        Designed to use:
        threads per block : 32
        blocks per grid   : NORMBLOCKS
        """
        
        j = cuda.grid(1)
        
        if j < N :
        
            s=0
            for i in range(0,N):
                s+=new_pi[i,j]
            
            for i in range(0,N):
                new_pi[i,j]/=s
            
            
    @cuda.jit
    def U_V_calc(mu,nu,new_U,new_V):
        """
        Gives the new values for U and V given the matrices mu and nu.
        
        Designed to use:
        threads per block : 32
        blocks per grid   : UVBLOCKS
        """
        
        ind=cuda.grid(1)
        
        if ind < N*L :
            
            l=ind//N
            i=ind%N
            
            t_l=(1+l)*D
            
            new_U[i,l]=mu[i,t_l]
            new_V[i,l]=nu[i,t_l]
            
            
    @cuda.jit
    def updt_pi(new_pi,pi):
        """
        Takes on new_pi and sets it on matrix pi.
        
        Designed to use:
        threads per block : 32
        blocks per grid   : PIINITBLOCKS
        """
        
        ind=cuda.grid(1)
        
        if ind < N*N :
            
            i=ind//N
            j=ind%N
            
            pi[i,j]=new_pi[i,j]
    
    
    ##### Set variables to GPU #####
    
    
    Y=cuda.to_device(Y)
    p_0=cuda.to_device(p_0)
    pi=cuda.to_device(pi)
    B=cuda.to_device(B)
    U=cuda.to_device(U)
    V=cuda.to_device(V)
    
    mu=cuda.device_array((N,T),np.float64)
    nu=cuda.device_array((N,T),np.float64)
    
    new_pi=cuda.device_array((N,N),np.float64)
    
    
    ##### Cycle #####
 

    for it in range(iters):
        
        if report_time:
            cuda.synchronize()
            t=time.perf_counter()
            print("")
            print("--- Iteration",it,"---")
        
        mu_nu_cuda[2*MUBLOCKS,block_size](Y,p_0,pi,B,U,V,mu,nu)
        
        if report_time:
            cuda.synchronize()
            print("munu :",time.perf_counter()-t)
            t=time.perf_counter()
        
        pi_zero_init[PIINITBLOCKS,32](new_pi)
        pi_calc[n_blocks,block_size](Y,pi,B,mu,nu,new_pi)
        pi_normalize[NORMBLOCKS,32](new_pi)
        updt_pi[PIINITBLOCKS,32](new_pi,pi)
        
        if report_time:
            cuda.synchronize()
            print("pi   :",time.perf_counter()-t)
            t=time.perf_counter()
        
        B_zero_init[BINITBLOCKS,32](B)
        B_calc[n_blocks,block_size](Y,mu,nu,B,p_0)
        B_normalize[NORMBLOCKS,32](B)
        
        if report_time:
            cuda.synchronize()
            print("B    :",time.perf_counter()-t)
        
        if L>0:
            U_V_calc[UVBLOCKS,32](mu,nu,U,V)
        

    p_0=p_0.copy_to_host()
    pi=pi.copy_to_host()
    B=B.copy_to_host()
    U=U.copy_to_host()
    V=V.copy_to_host()
    
    warnings.simplefilter('default')
    
    return(N,M,p_0,pi,B,D,L,U,V)

