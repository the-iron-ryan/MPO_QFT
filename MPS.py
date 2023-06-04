
from TensorHelpers import *

import numpy as np

import quimb as qu
import quimb.tensor as qtn
import quimb.tensor.tensor_core as qtc

'''
MPS class
'''
class MPS:
    def __init__(self, N):
        self.N = N
        
    '''
    Creates a MPS from a given function f
    
    Parameters:
    ----------
    fs: list of function
        Set of functions in the form e^ix to be used to create the MPS
        
    start: float
        Starting point of the function
        
    end: float
        End point of the function
        
    max_bond: int
        Maximum bond dimension of the MPS
    
    cutoff: float
        cutoff SVD value for MPS compression
    '''
    def create_MPS(self, fs, start, stop, max_bond=-1, cutoff=1e-10):
        mps_list = []
        for f in fs:
            input = np.arange(start, stop, step=1/(2**self.N))
            state = f(input)
            
            cur_mps = qtn.MatrixProductState.from_dense(state, dims=[2]*self.N, method='svd', max_bond=max_bond, cutoff=cutoff, absorb='right')
            mps_list.append(cur_mps)
          
        final_mps = mps_list[0] 
        for mps in mps_list[1:]:
            final_mps += mps
            
        return final_mps