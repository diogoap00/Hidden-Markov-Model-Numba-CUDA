# -*- coding: utf-8 -*-
"""
Created on Mon May 15 23:06:17 2023

@author: Diogo Pereira
"""

import numpy as np
from hmmdisc import basic as hmmdb


"""
The hmmdisc package uses tuple of parameters representing the parameters of the HMM.

For the simple set of parameters (with no U and V matrices as in the paper) we have 
a tuple with 5 elements such that:
element 0: should be a positive integer, N, denoting the number of hidden states
element 1: should be a positive integer, M, denoting number of possible values for the observations.
element 2: p_0, the initial condition for the hidden Markov process, 1 dim array
element 3: pi the transition matrix of the hidden Markov process, 2 dim array
element 4: matrix B, (M x N) matrix, denoting the emission probabilities for the observations.
"""

#  Example set of parameters

N=3
M=3

p_0=np.array([1.,0.,0.])

pi=np.array([[0.90, 0.05, 0.05],
             [0.05, 0.90, 0.05],
             [0.05, 0.05, 0.90]])

B=np.array([[0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90]])

params=(N,M,p_0,pi,B)


# use the function hmmdb.check_fin_params(params) to check if the parameters are valid
# It will raise an error if an issue is found

hmmdb.check_fin_params(params)


# Use the function hmmdb.sim_fin(T,params) to simulate a random sample path according
# to parameters params. Here T is the lenth of the simulated series

T=20000

Y=hmmdb.sim_fin(T,params)


# Now we initialize some initial parameters for the Baum-Welch algorithm

N=3
M=3

p_0=np.ones(N)/N

pi=np.array([[0.8, 0.1, 0.1],
             [0.1, 0.8, 0.1],
             [0.1, 0.1, 0.8]])

B=np.array([[0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]])

params_start=(N,M,p_0,pi,B)
hmmdb.check_fin_params(params_start)


# To do an iteration of the Baum-Welch algorithm use the command

params_calc=hmmdb.fin_em_it(Y,params_start)


# To do the iteration multiple times you may use

iters=20
for it in range(iters):
    params_calc=hmmdb.fin_em_it(Y,params_calc)





















