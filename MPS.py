
from TensorHelpers import *

import quimb as qu
import quimb.tensor as qtn
import quimb.tensor.tensor_core as qtc

'''
MPS class
'''
class MPS:
    def __init__(self, N):
        self.N = N
        
    def create_MPS(self, f):
        pass