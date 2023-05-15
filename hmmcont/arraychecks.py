#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 20:37:06 2022

@author: diogoap00

This is a module to check that a numpy array is in a correct format, 
for instance check if it's a probability array or a positive integer.
Functions raise an error if they find a problem.
"""


import numpy as np


TYPES=["float16","float32","float64","float128","float"]
TYPES_INT=["int16","int32","int64","int128","int"]


def pos_int(N,name="N"):
    """
    Checks if N is a positive integer
    """
    
    if not isinstance(N,int):
        raise TypeError(name+" is not an integer")
    if not N>0:
        raise ValueError(name+" is not positive")
        
        
def non_neg(N,name="N"):
    """
    Checks if N is a non-negative integer
    """
    
    if not isinstance(N,int):
        raise TypeError(name+" is not an integer")
    if not N>=0:
        raise ValueError(name+" is not non-negative")
    
    
def vector(vec,size=None,name="vec"):
    """
    Checks if vec is a 1-dim array with given size and has no nan values.
    if size=None function skips the size check. 
    """
    
    if not isinstance(vec,np.ndarray):
        raise TypeError(name+" is not a np.ndarray")
    if vec.ndim!=1:
        raise ValueError(name+" is not 1 dimensional")
    if not vec.dtype in TYPES:
        raise ValueError(name+" is not a valid float type")
    if not size==None:
        non_neg(size,name="Input size")
        if not vec.shape==(size,):
            raise ValueError(name+" is not the correct size")
    if True in np.isnan(vec):
        raise ValueError(name+" has np.nan values")
    
    
def prob_vector(vec,size=None,name="vec",tol=1e-10):
    """
    Checks if vec is a 1-dim array with given size, has no nan values, and 
    defines a probability distribution (sums to 1).
    if size=None function skips the size check.
    tol defines the tolerance to the error of vec sum to 1.
    """
    
    vector(vec,size=size,name=name)
    if True in (vec<0) or True in (vec>1):
        raise ValueError(name+" does not have all values between 0 and 1")
    if not np.abs(np.sum(vec)-1)<tol:
        raise ValueError(name+" does not sum to 1")
    

def matrix(mat,shape=None,name="mat"):
    """
    Checks if mat is a 2-dim array with given shape and has no nan values.
    if shape=None function skips the shape check.
    """
    
    if not isinstance(mat,np.ndarray):
        raise TypeError(name+" is not a np.ndarray")
    if mat.ndim!=2:
        raise ValueError(name+" is not 2 dimensional")
    if not mat.dtype in TYPES:
        raise ValueError(name+" is not a valid float type")
    if not shape==None:
        if not mat.shape==shape:
            raise ValueError(name+" is not the correct shape")
    if True in np.isnan(mat):
        raise ValueError(name+" has np.nan values")
        
        
def trans_matrix(mat,shape=None,name="mat",tol=1e-10):
    """
    Checks if mat is a 2-dim array with given shape, has no nan values, and 
    it's columns define a probability distribution (sum to 1).
    if shape=None function skips the size check.
    tol defines the tolerance to the error of the columns sum to 1.
    """
    
    matrix(mat,shape=shape,name=name)
    if True in (mat<0) or True in (mat>1):
        raise ValueError(name+" must have values between 0 and 1")
    if False in (np.abs(np.sum(mat,axis=0)-1)<tol):
        raise ValueError(name+" columns do not sum to 1")


def vector_index(vec,size=None,name="vec"):
    """
    Checks if vec is a 1-dim  non-negative integer array with given size and has no nan values.
    if size=None function skips the size check. 
    """
    
    if not isinstance(vec,np.ndarray):
        raise TypeError(name+" is not a np.ndarray")
    if vec.ndim!=1:
        raise ValueError(name+" is not 1 dimensional")
    if not vec.dtype in TYPES_INT:
        raise ValueError(name+" is not a valid int type")
    if not size==None:
        non_neg(size,name="Input size")
        if not vec.shape==(size,):
            raise ValueError(name+" is not the correct size")
    if True in np.isnan(vec):
        raise ValueError(name+" has np.nan values")
    if (vec<0).any():
        raise ValueError(name+" has negative values")
    
    