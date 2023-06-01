'''
Tensor Helper Class
'''
class TensorHelpers:
    
    @classmethod
    def getTag(cls, row, col):
        '''
        Converts tuple of row, col to a string
        '''
        return f'({row:.1f}, {col:.1f})'
        
    @classmethod 
    def getRowColTagFromTensor(cls, tensor):
        '''
        Gets the row,col tag of a tensor
        '''
        for tag in tensor.tags:
            if tag[0] == '(':
                return tag
    @classmethod
    def getTensorIndices(cls, tensor):
        '''
        Gets a list of tensor indices wrapped around Index objects
        '''
        indices = []
        for i in tensor.inds:
            indices.append(i)
        return indices
    
    @classmethod 
    def getTagRowCol(cls, tag):
        '''
        Converts a tag to a tuple of row, col
        '''
        return (float(val) for val in tag[1:-1].split(','))

    @classmethod 
    def getIndexRowCol(cls, index):
        '''
        Converts an index to a tuple of row, col
        '''
        return TensorHelpers.getTagRowCol(str(index))
    
    @classmethod
    def getTensorRowCol(cls, tensor):
        '''
        Gets the row, col of a tensor
        '''
        tensor_row_col_tag = TensorHelpers.getRowColTagFromTensor(tensor)
        return TensorHelpers.getTagRowCol(tensor_row_col_tag)