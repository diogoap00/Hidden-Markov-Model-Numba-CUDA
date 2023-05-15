# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 15:02:04 2023

@author: Diogo Pereira


This a library to work with hidden markov models in continuous observation space. 
It includes simulation functions, likelihood functions, and 
expectation maximization related functions.

The gpu module implements the basic.arp_em_it in a cuda implementation.
"""


import numpy as np
from numba import cuda, float64, int64
from hmmcont import basic as hmmb
from hmmcont import arraychecks as ac
import time
import math
import warnings




def arp_cuda_em_it(Y,params,iters=1,block_size=32,n_blocks=128,report_time=False):
    """
    Equivalent function to basic.arp_em_it but uses cuda (Nvidia GPU).
    Y should be a np.array of data and params as basic.check_arp_params.
    
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
    
    ac.vector(Y,name="Y")
    if len(Y)<M: 
        raise ValueError("len(Y) must be at least M")
    hmmb.check_arp_params(params)
    
    T=len(Y)-M+1
    
    if not ( 1+D*L < T and T <= 1+D*(L+1) ):
        raise ValueError("Parameters must satisfy 1+D*L < len(Y)-M <= 1+D*(L+1)")
        
    if block_size < N :
        raise ValueError("block_size must be greater or equal to N (default is 32)")


    ##### Functions #####
    
    # constants to be used
    
    SQPI=1/math.sqrt(2*math.pi)
    
    SEGBLOCKS=int(block_size//N)
    MUBLOCKS=int(np.ceil((L+1)/SEGBLOCKS))
    
    NSUMSSIZE=2**math.ceil(math.log(N,2)-1)
    if N==1: NSUMSSIZE=1
    
    PIINITBLOCKS=int(np.ceil(N**2/(32)))
    SYSTINITBLOCKS=int(np.ceil(N*M*M/(32)))
    INDINITBLOCKS=int(np.ceil(N*M/(32)))
    SIGMAINITBLOCKS=int(np.ceil(N/(32)))
    
    NORMBLOCKS=int(np.ceil(N/(32)))
    UVBLOCKS=int(np.ceil(N*L/(32)))
    
    MP1=M+1
    
    
    @cuda.jit
    def G_calc(Y,A,sigma,G):
        """
        Calculates the (N x T) matrix G (\Gamma in the paper).
        Takes Y and the parameters as input
        
        Designed to use:
        threads per block : block_size
        blocks per grid   : n_blocks
        """  
        
        thread_ind=cuda.grid(1)
        
        for ind in range(thread_ind, N*T, block_size*n_blocks):
            
            k = ind // N
            i = ind % N
            
            pred=A[i,0]
            for m in range(1,M):
                pred=pred+Y[k+M-m-1]*A[i,m]
            
            G[i,k]=1/sigma[i]*SQPI*math.exp(-1/2*((Y[k+M-1]-pred)/sigma[i])**2)
    
    
    @cuda.jit
    def mu_nu_cuda(Y,p_0,pi,U,V,G,mu,nu):
        """
        Calculates the mu nu matrices, given some parameters, the dataset Y
        and calculated matrix G from G_calc function.
        
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
        
                    tmp_last[i,0]=G[i,0]*p_0[i]
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
                            
                        tmp[i,seg]=G[i,index]*tmp[i,seg]
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
                        
                        gbg_tmp[i,seg]=G[i,index]*tmp_last[i,seg]

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
    def syst_zero_init(A_syst):
        """
        Function to initialize a (N x M x M) tensor to zero
        
        Designed to use:
        threads per block : 32
        blocks per grid   : SYSTINITBLOCKS
        """
        
        ind=cuda.grid(1)
        
        if ind < N*M*M :
            
            i=ind%N
            jk=ind//N
            j=jk%M
            k=jk//M
            
            A_syst[i,j,k]=0
            
            
    @cuda.jit
    def ind_zero_init(A_ind):
        """
        Function to initialize a (N x M) matrix to zero
        
        Designed to use:
        threads per block : 32
        blocks per grid   : INDINITBLOCKS
        """
        
        ind=cuda.grid(1)
        
        if ind < M*N:
            
            j=ind//N
            i=ind%N
            
            A_ind[i,j]=0
            
            
    @cuda.jit
    def sigma_zero_init(sigma_new):
        """
        Function to initialize a (N x 1) vector to zero
        
        Designed to use:
        threads per block : 32
        blocks per grid   : SIGMAINITBLOCKS
        """
        
        ind=cuda.grid(1)
        
        if ind < N:
            
            sigma_new[ind]=0
            
            
    @cuda.jit
    def pi_calc(Y,pi,G,mu,nu,new_pi):
        """
        Calculates the new matrix pi, given G, mu and nu. Matrix new_pi will not be
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
                
                mat_tmp_sum[i,seg]=0
                cuda.syncthreads()
                
                gbg_tmp[i,seg]=nu[i,k+1]*G[i,k+1]
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
    def syst_calc(Y,mu,nu,A_syst,A_ind):
        """
        Calculates the system relating to parameters in A. Does not solve
        the system requires an additional function. Requires A_syst and A_ind
        initialized to zero.
        
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
        
        A_syst_tmp=cuda.shared.array((N,M,M,SEGBLOCKS), dtype=float64)
        A_ind_tmp=cuda.shared.array((N,M,SEGBLOCKS), dtype=float64)
        
        if seg<SEGBLOCKS:
            
            for j in range(0,M):
                for m in range(0,M):
                    A_syst_tmp[i,j,m,seg]=0
                A_ind_tmp[i,j,seg]=0
            
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

                cond_vec[i,seg]=cond_vec[i,seg]/cond_vec_sum[0,seg]
                
                A_ind_tmp[i,0,seg] += Y[k+M-1]*cond_vec[i,seg]
                A_syst_tmp[i,0,0,seg] += cond_vec[i,seg]
                for m in range(1,M):
                    A_syst_tmp[i,0,m,seg] += Y[k+M-1-m]*cond_vec[i,seg]
                
                for j in range(1,M):
                    A_ind_tmp[i,j,seg] += Y[k+M-1-j]*Y[k+M-1]*cond_vec[i,seg]
                    A_syst_tmp[i,j,0,seg] += Y[k+M-1-j]*cond_vec[i,seg]
                    for m in range(1,M):
                        A_syst_tmp[i,j,m,seg] += Y[k+M-1-j]*Y[k+M-1-m]*cond_vec[i,seg]

            for j in range(0,M):
                cuda.atomic.add(A_ind, (i,j), A_ind_tmp[i,j,seg])
                for m in range(0,M):
                    cuda.atomic.add(A_syst, (i,j,m), A_syst_tmp[i,j,m,seg])
                
                
    @cuda.jit
    def solve_syst(A_syst,A_ind,A_new):
        """
        Solves the linear systems given by A_syst and A_ind calculated in syst_calc
        
        Designed to use:
        threads per block : (M x M+1)
        blocks per grid   : N
        """        
        
        j = cuda.threadIdx.x
        m = cuda.threadIdx.y
        b = cuda.blockIdx.x
        
        syst_tmp=cuda.shared.array((M,MP1), dtype=float64)
        vec_tmp=cuda.shared.array((MP1,), dtype=float64)
        c_tmp=cuda.shared.array((M,), dtype=float64)
        success=cuda.shared.array((1,), dtype=int64)
        ind=cuda.shared.array((1,), dtype=int64)
        
        if m < M :
            syst_tmp[j,m]=A_syst[b,j,m]
        if m == M :
            syst_tmp[j,m]=A_ind[b,j]
        cuda.syncthreads()
        
        success[0]=0
        
        for k in range(0,M):
            
            if j==0 and m==0 :
                for l in range(k,M):
                    if math.fabs(syst_tmp[l,k]) > 10e-6 :
                        ind[0]=l
                        break
                    if l==M-1 :
                        success[0]=1
            cuda.syncthreads()
            
            if success[0]==1 :
                if j==0 and m==0 :
                    success[0]==0
                cuda.syncthreads()
                continue
            
            if j==0 :
                vec_tmp[m]=syst_tmp[ind[0],m]
                syst_tmp[ind[0],m]=syst_tmp[k,m]
                syst_tmp[k,m]=vec_tmp[m]
            cuda.syncthreads()
            
            if j==k :
                syst_tmp[j,m]=syst_tmp[j,m]/syst_tmp[j,j]
            
            else:
                if m==0 :
                    c_tmp[j]=syst_tmp[j,k]/vec_tmp[k]
                cuda.syncthreads()
                
                syst_tmp[j,m]=syst_tmp[j,m]-vec_tmp[m]*c_tmp[j]
        
        if m==M :
            A_new[b,j]=syst_tmp[j,m]
        
                
    @cuda.jit
    def sigma_calc(Y,mu,nu,A_new,sigma_new,norms,p_0):
        """
        Calculates the parameters for sigma. Requires the extra step
        math.sqrt(sigma_new[i]/norms[i]) to get the new sigma parameters. 
        The vectors sigma_new and norms should be initialized to zero.
        Also updates p_0.
        
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
        
        errors_tmp=cuda.shared.array((N,SEGBLOCKS), dtype=float64)
        errors_tmp_sum=cuda.shared.array((N,SEGBLOCKS), dtype=float64)
        norms_tmp=cuda.shared.array((N,SEGBLOCKS), dtype=float64)
        
        if seg<SEGBLOCKS:
            
            errors_tmp_sum[i,seg]=0
            norms_tmp[i,seg]=0
            
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
        
                cond_vec[i,seg]=cond_vec[i,seg]/cond_vec_sum[0,seg]
                
                errors_tmp[i,seg]=Y[k+M-1]-A_new[i,0]
                for m in range(1,M):
                    errors_tmp[i,seg] += -A_new[i,m]*Y[k+M-1-m]
                
                errors_tmp_sum[i,seg] += math.pow(errors_tmp[i,seg],2)*cond_vec[i,seg]
                norms_tmp[i,seg] += cond_vec[i,seg]
                
                if k==0 :
                    p_0[i]=cond_vec[i,0]
            
            cuda.atomic.add(sigma_new, i, errors_tmp_sum[i,seg])
            cuda.atomic.add(norms, i, norms_tmp[i,seg])
            
            
    @cuda.jit
    def sigma_sqrt_divide(sigma_new,norms):
        """
        Dividides element-wise a (N x 1) vector to a (N x 1) vector 
        and applies the square-root. Saves the result in sigma_new.
        
        Designed to use:
        threads per block : 32
        blocks per grid   : SIGMAINITBLOCKS
        """
        
        ind=cuda.grid(1)
        
        if ind < N:
            
            sigma_new[ind]=math.sqrt(sigma_new[ind]/norms[ind])
            
    
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
    A=cuda.to_device(A)
    sigma=cuda.to_device(sigma)
    U=cuda.to_device(U)
    V=cuda.to_device(V)
    
    G=cuda.device_array((N,T),np.float64)
    mu=cuda.device_array((N,T),np.float64)
    nu=cuda.device_array((N,T),np.float64)
    
    new_pi=cuda.device_array((N,N),np.float64)
    
    A_syst=cuda.device_array((N,M,M),np.float64)
    A_ind=cuda.device_array((N,M),np.float64)
    
    norms=cuda.device_array((N,),np.float64)
    
    
    ##### Cycle #####
    
    
    for it in range(iters):
        
        if report_time:
            cuda.synchronize()
            t=time.perf_counter()
            print("")
            print("--- Iteration",it,"---")
        
        G_calc[n_blocks,block_size](Y,A,sigma,G)
        mu_nu_cuda[2*MUBLOCKS,block_size](Y,p_0,pi,U,V,G,mu,nu)
        
        if report_time:
            cuda.synchronize()
            print("munu :",time.perf_counter()-t)
            t=time.perf_counter()
        
        pi_zero_init[PIINITBLOCKS,32](new_pi)
        pi_calc[n_blocks,block_size](Y,pi,G,mu,nu,new_pi)
        pi_normalize[NORMBLOCKS,32](new_pi)
        updt_pi[PIINITBLOCKS,32](new_pi,pi)
        
        if report_time:
            cuda.synchronize()
            print("pi   :",time.perf_counter()-t)
            t=time.perf_counter()
        
        syst_zero_init[SYSTINITBLOCKS,32](A_syst)
        ind_zero_init[INDINITBLOCKS,32](A_ind)
        syst_calc[n_blocks,block_size](Y,mu,nu,A_syst,A_ind)
        solve_syst[N,(M,M+1)](A_syst,A_ind,A)
        
        if report_time:
            cuda.synchronize()
            print("syst :",time.perf_counter()-t)
            t=time.perf_counter()
        
        sigma_zero_init[SIGMAINITBLOCKS,32](sigma)
        sigma_zero_init[SIGMAINITBLOCKS,32](norms)
        sigma_calc[n_blocks,block_size](Y,mu,nu,A,sigma,norms,p_0)
        sigma_sqrt_divide[SIGMAINITBLOCKS,32](sigma,norms)
        
        if report_time:
            cuda.synchronize()
            print("sig  :",time.perf_counter()-t)
        
        if L>0:
            U_V_calc[UVBLOCKS,32](mu,nu,U,V)
    

    p_0=p_0.copy_to_host()
    pi=pi.copy_to_host()
    A=A.copy_to_host()
    sigma=sigma.copy_to_host()
    U=U.copy_to_host()
    V=V.copy_to_host()
    
    warnings.simplefilter('default')
    
    return(N,M,p_0,pi,A,sigma,D,L,U,V)
    
