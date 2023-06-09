
from qiskit.circuit.library.standard_gates import CPhaseGate, PhaseGate

import quimb as qu
import quimb.tensor as qtn
import quimb.tensor.tensor_core as qtc

import numpy as np
import math

# Local imports
from TensorHelpers import *
from Index import *
    
    
'''
Class to Handle QFT Tensor Network
'''
class QFT:
    def __init__(self, N):
        self.tn = qtn.TensorNetwork([])
        self.N = N

    
    def _getClosestIndex(self, tensor, row, col):
        '''
        Gets the closest index of a tensor to a given row and col
        '''

        smallest_dist = math.inf
        closest_index = None

        # Get only indices that are not connected to other tensors
        unconnected_indices = filter(lambda i: '->' not in i, tensor.inds)
        for i in unconnected_indices:
            cur_row, cur_col = TensorHelpers.getIndexRowCol(i)
            cur_dist = math.sqrt((cur_row - row)**2 + (cur_col - col)**2)
            if cur_dist < smallest_dist:
                smallest_dist = cur_dist
                closest_index = i
        
        if closest_index is None:
            raise Exception('Could not find closest index')
        
        return closest_index


    def getTensorFromIndex(self, index):
        '''
        Gets a tensor from an index
        '''
       
        # Convert to string if necessary 
        if isinstance(index, Index):
            index = str(index)
        
        if index in self.tn.ind_map:
            matching_tensors = list(map(self.tn.tensor_map.get, self.tn.ind_map[index]))
            tensor = matching_tensors[0]
            return tensor
        else:
            return None
        
    def getTensorFromRowCol(self, row, col):
        '''
        Gets a tensor from a row and col
        '''
        tag = TensorHelpers.getTag(row, col)
        return self.getTensorFromTag(tag)
    
    def getTensorFromTag(self, tag):
        '''
        Gets a tensor from a tag
        '''
        if tag in self.tn.tag_map:
            # Complicated way of getting the tensor
            matching_tensors = list(map(self.tn.tensor_map.get, self.tn.tag_map[tag]))

            # return the first matching tensor
            tensor = matching_tensors[0]
            return tensor
        else:
            return None
        
    def getTensorsFromIndex(self, index):
        '''
        Gets all tensors from an index
        '''
        if isinstance(index, Index):
            index = str(index)
        
        if index in self.tn.ind_map:
            matching_tensors = list(map(self.tn.tensor_map.get, self.tn.ind_map[index]))
            return matching_tensors
        else:
            return None
        
    def get_sorted_tensors(self):
        '''
        Gets a list of all tensors sorted by row and col
        '''
        
        # Sort by row and col in tuple form
        def sort_func(t):
            row, col = TensorHelpers.getTensorRowCol(t)
            return (row, col)
        
        MPO_tensors = self.tn.tensors
        return sorted(MPO_tensors, key=sort_func)
        

    def getIndices(self, tensor_row, tensor_col, bonds=[]):
        '''
        Gets corresponding indices for a tensor for each of its bonds.
        Also updates the tensor network with any matching bonds

        Parameters
        ----------
        tensor_row : int
            The row of the tensor within the quantum circuit.
            Row 0 is the top row.
        tensor_col : int
            The column of the tensor within the quantum circuit.
            Column 0 is the leftmost column.
        bonds : list of tuples
            A list of tuples representing the bonds indices the 
            tensor should connect to.

            ex: [(0, 1), (1, 0)] would connect the tensor to the
            tensor at row 0, col 1 and the tensor at row 1, col 0

        '''

        indices = []

        # Get the index based on row, col
        tensor_tag = TensorHelpers.getTag(tensor_row, tensor_col)

        # Loop through all our bonds and make idex connections
        for b in bonds:
            # Get our current bond row and col. Clamp it between the bounds of the tensor network
            bond_row, bond_col = b
            bond_tag = TensorHelpers.getTag(bond_row, bond_col)

            # Current bond index
            bond_index = Index(bond_tag, tensor_tag)
            indices.append(str(bond_index))

        return indices


    def add_hadamard(self, row, col):
        """
        Get the MPO for the Hadamard gate and place it at the given row and col

        Parameters
        ----------
        row : int
            The row of the tensor within the quantum circuit.
            Row 0 is the top row.
        col : int
            The column of the tensor within the quantum circuit.
            Column 0 is the leftmost column.
        """
        return self.tn.add(self.get_hadamard(row, col))
        
    def get_hadamard(self, row, col):
        inds = self.getIndices(row, col, bonds=[
            (row, col-1), 
            (row, col+1)
        ])
        return qtn.Tensor(qu.hadamard(), inds=inds, tags=['H', TensorHelpers.getTag(row, col)])
        

    def add_phase_MPO(self, start_row, end_row, col):
        '''
        Returns a tensor network MPO representing the controlled phase gates

        Parameters
        ----------
        start_row : int
            the starting row of the phase MPO
        '''
        for t in self.get_phase_MPO(start_row, end_row, col):
            self.tn.add(t)

    def get_phase_MPO(self, start_row, end_row, col):
        '''
        Returns the tensors for an MPO representing the controlled phase gates

        Parameters
        ----------
        start_row : int
            the starting row of the phase MPO
        '''

        tensors = []
        
        # Case where we have a single site. Only return a hadamard
        if start_row == end_row:
            tensors.append(self.add_hadamard(start_row, col))
            return
        
        # Internal function to get the data for a order 4 phase gate
        def get_cphase_data(phase):
            return np.array([
                [ qu.identity(2), np.zeros((2,2))],
                [ np.zeros((2,2)), qu.phase_gate(phase)]
            ])

        # Define our control (copy tensor) in the form:
        # [ 
        #   [1, 0]  [0, 0] 
        #   [0, 0], [0, 1] 
        # ]
        # (MPO-QFT paper equation 53)
        copy_tensor_data = np.array([ 
             [[1, 0], 
              [0, 0]],
             
             [[0, 0], 
              [0, 1]]
        ])
        copy_tensor = qtn.Tensor(
            data=copy_tensor_data, 
            inds = self.getIndices(start_row, col, bonds=[
                (start_row+1, col), # control bond
                (start_row, col-1), # left row vector bond
                (start_row, col+1), # right col vector bond
            ]),
            tags = ['C', TensorHelpers.getTag(start_row, col)])

        tensors.append(copy_tensor)
        
        # Apply each of our controlled phase gates in the form:
        # [
        #   [1, 0]  [0, 0]
        #   [0, 1], [0, 0],
        #   [0, 0]  [1, 0]
        #   [0, 0], [0, e^iπ)]
        # ]
        # (MPO-QFT paper equation 53)
        phase_count = 1
        for i in range(start_row+1, end_row-1):
            phase_denom = (2**(phase_count))
            phase = -np.pi/phase_denom
            inds = self.getIndices(i, col, bonds=[
                (i-1, col), # top control bond (row)
                (i+1, col), # bottom control bond (col)
                (i, col-1), # left row vector bond
                (i, col+1), # right col vector bond
            ])

            tensors.append(qtn.Tensor(
                data=get_cphase_data(phase),
                inds=inds,
                tags=['P', f'$\pi$/{phase_denom}', TensorHelpers.getTag(i,col)]))

            phase_count += 1

        # Apply last phase gate in the form:
        # [ 
        #   [1, 0]
        #   [0, 1],
        #   [1, 0]
        #   [0, e^iπ)] 
        # ]
        # (MPO-QFT paper equation 53)
        last_phase_denom = (2**(phase_count))
        last_phase = -np.pi/last_phase_denom
        last_phase_gate_inds = self.getIndices(end_row-1, col, bonds=[
            (end_row-2, col),
            (end_row-1, col-1),
            (end_row-1, col+1)
        ])

        tensors.append(qtn.Tensor(
            data=np.array([qu.identity(2), qu.phase_gate(last_phase)]).reshape(2,2,2),
            inds=last_phase_gate_inds,
            tags=['P', f'$\pi$/{last_phase_denom}', TensorHelpers.getTag(end_row-1, col)]))
        return tensors
    def merge(self):
        '''
        Merges all unconnected indices of phase gates in our tensor network
        '''

        # Filter out all indices that are only connected to one tensor
        unconnected_indices_strs = filter(lambda i: len(self.tn.ind_map[i]) == 1, self.tn.ind_map.keys())

        # Convert the indices to Index objects
        unconnected_indices = list(map(lambda i: Index(i), unconnected_indices_strs))

        reindex_map = {}

        # Loop through each index and check if we have any intersecting indices
        for i in range(len(unconnected_indices)):
            cur_index = unconnected_indices[i]
            for j in range(i+1, len(unconnected_indices)):
                other_index = unconnected_indices[j]
                intersections = cur_index.getIntersections(other_index)
                if intersections is not None and len(intersections) > 0:
                    cur_tensor = self.getTensorFromIndex(str(cur_index))
                    other_tensor = self.getTensorFromIndex(str(other_index))

                    # Make sure we have two different tensors. Don't want to merge edges of the same tensor
                    if cur_tensor is not None and other_tensor is not None and cur_tensor != other_tensor:
                        # Get the indices we're merging on
                        cur_tensor_inds = TensorHelpers.getRowColTagFromTensor(cur_tensor)
                        other_tensor_inds = TensorHelpers.getRowColTagFromTensor(other_tensor)

                        # current remap index that we're going to merge into
                        cur_remap_index = Index(cur_tensor_inds, other_tensor_inds)

                        # Add the indices to our reindex map
                        reindex_map[str(cur_index)] = str(cur_remap_index)
                        reindex_map[str(other_index)] = str(cur_remap_index)


        # Finally, do an inplace remap of the indices
        self.tn.reindex(reindex_map, inplace=True)

    def create_circuit(self):
        '''
        Creates a tensor network circuit representing the quantum Fourier transform
        '''
        self.tn = qtn.TensorNetwork()
        
        row = 0
        col = 0
        while row < self.N-1 and col < 2*self.N-1:
            self.add_hadamard(row, col)
            self.add_phase_MPO(row, self.N, col+1)

            row += 1
            col += 2
        self.add_hadamard(row, col)

        # Important: merge all unconnected phase gates after we've built the circuit
        self.merge()
        
    def create_circuit_from_apply(self, max_bond_dim=-1, cutoff=1e-15, verbose=False, reverse=False):
        row = 0
        col = 0
        while row < self.N-1 and col < 2*self.N-1:
            # self.add_hadamard(row, col)
            phase_mpo_tensors = self.get_phase_MPO(row, self.N, col)
            phase_mpo_data = list(map(lambda t: t.data, phase_mpo_tensors))
            
            phase_mpo = qtn.MatrixProductOperator(phase_mpo_data)
            phase_mpo.compress_all(inplace=True, max_bond=max_bond_dim, cutoff=cutoff)

            row += 1
            col += 1
            
        
        # self.add_hadamard(row, col)

        # Important: merge all unconnected phase gates after we've built the circuit
        self.merge()
        

    def create_MPO(self, max_bond_dim=-1, cutoff=1e-15, verbose=False, reverse=False):
        '''
        Creates a matrix product operator representing the quantum Fourier transform
        
        ----------
        Parameters
        
        max_bond_dim: int
            The maximum bond dimension to use when zipping up the tensor network
        
        Returns
        mpo: qtn.MatrixProductOperator
            The matrix product operator representing the quantum Fourier transform
        '''

        self.create_circuit()
        self.zip_up(max_bond=max_bond_dim, cutoff=cutoff, verbose=verbose)
        
        # Do final global compression
        self.tn.compress_all(inplace=True, max_bond=max_bond_dim)
        
        if verbose:
            self.draw("Finished Zip Up")
        
        # Get the tensors in the correct order
        sorted_tensors = self.get_sorted_tensors()
        
        # Print out the tensors 
        if verbose:
            self.print_tensors()
        
        # Convert the tensors to numpy arrays 
        MPO_arrays = [t.data for t in sorted_tensors]
       
        mpo = qtn.MatrixProductOperator(MPO_arrays, shape='udlr', bond_name='z')
        # mpo.right_canonize()
        
        if reverse:
            mpo_tags = list(mpo.tag_map.keys())
            reversed_mpo_tags = list(reversed(mpo_tags))
            reverse_tag_map = { mpo_tags[i]: reversed_mpo_tags[i] for i in range(len(reversed_mpo_tags))}
            
            mpo.retag(reverse_tag_map, inplace=True)

            reversed_b_ind_map = { f'b{i}': f'b{self.N-1-i}' for i in range(self.N)}
            mpo.reindex(reversed_b_ind_map, inplace=True)
            reversed_k_ind_map = { f'k{i}': f'k{self.N-1-i}' for i in range(self.N)}
            mpo.reindex(reversed_k_ind_map, inplace=True)

        if verbose:
            mpo.draw(color=['Q', 'P', 'C', 'H'], show_inds='bond-size', show_tags=True, figsize=(20, 20))

        return mpo
        
    def getValidTensorsInRange(self, row_range, col_range):
        '''
        Gets all valid tensors in a given range
        '''
        valid_tensors = []
        for row in np.arange(row_range[0], row_range[-1]+0.5, 0.5):
            for col in np.arange(col_range[0], col_range[-1]+0.5, 0.5):
                tensor = self.getTensorFromTag(TensorHelpers.getTag(row, col))
                if tensor is not None:
                    valid_tensors.append(tensor)
        
        return valid_tensors
    def contract(self, tensor_tags):
        '''
        Contracts all tensors in a given range
        '''
        self.tn.contract(tensor_tags, inplace=True)
        
        return tensor_tags[0]
    
    def contract_tensors_in_range(self, row_range, col_range):
        ''''
        Contracts all tensors in a given row/col range
        '''
        valid_tensors = self.getValidTensorsInRange(row_range, col_range)
        valid_tensor_tags_to_contract = [TensorHelpers.getRowColTagFromTensor(tensor) for tensor in valid_tensors]
        
        # Contract our tensors and fetch the tensor plus its index tag
        contracted_tensor_tag = self.contract(valid_tensor_tags_to_contract)
        contracted_tensor = self.getTensorFromTag(contracted_tensor_tag)    
        
        return contracted_tensor, contracted_tensor_tag
        
    def getSVDIndices(self, tensor, tensor_row, direction='up'):
        '''
        Gets the SVD indices of tensors in a given direction
        
        Parameters
        ----------
        tensor: qtn.Tensor
            tensor to get the index of
            
        direction: str
            direction to get the index of. Can be 'up' or 'down'
        '''
        
        indices_in_dir = []
        
        tensor_indices = TensorHelpers.getTensorIndices(tensor)
        for index in tensor_indices:
            connected_tensors = self.getTensorsFromIndex(index)
            if len(connected_tensors) != 2:
                continue
            
            neighbor_tensor = connected_tensors[0] if connected_tensors[0] != tensor else connected_tensors[1]
            neighbor_tensor_row, neighbor_tensor_col = TensorHelpers.getTensorRowCol(neighbor_tensor)
            if direction == 'up':
                if neighbor_tensor_row < tensor_row:
                    indices_in_dir.append(index)
            elif direction == 'down':
                if neighbor_tensor_row > tensor_row:
                    indices_in_dir.append(index)
                    
        return indices_in_dir
            
    
    def zip_up(self, max_bond=-1, cutoff=1e-15, verbose=False):
        '''
        Performs the zip-up algorithm on the phase gates in our tensor network
        '''
        
        end_row = 1
        end_col = 2*self.N-3
        
        col = 1
        col_radius = 2.5
       
        while True: 
            # Do zip up
            for row in range(self.N-1, end_row, -1):
                contracted_tensor, contracted_tensor_tag = self.contract_tensors_in_range([row, row+0.5], [col-col_radius, col+col_radius])
               
                # Shift the tag of the contracted tensor up by 0.5 
                contracted_tag_row, contracted_tag_col = TensorHelpers.getTagRowCol(contracted_tensor_tag)
                new_contracted_tensor_tag = TensorHelpers.getTag(contracted_tag_row, contracted_tag_col - 0.5)
                contracted_tensor.retag({contracted_tensor_tag: new_contracted_tensor_tag}, inplace=True)
                contracted_tensor_tag = new_contracted_tensor_tag
               
                if verbose: 
                    self.draw(f"Zipping up: {row}, {col}") 
                
                # Get the indices we're going to SVD on
                left_inds = [str(index) for index in self.getSVDIndices(contracted_tensor, row, 'up')]
                right_inds = list(filter(lambda t: t not in left_inds, TensorHelpers.getTensorIndices(contracted_tensor)))
                
                absorb = "left"
                left_tags = [TensorHelpers.getTag(row-0.5, col+1), f'UxS[{row}]', 'T']
                right_tags = [TensorHelpers.getTag(row, col+1), f'V[{row}]']
                
                self.tn.replace_with_svd(
                    [contracted_tensor_tag], 
                    left_inds=left_inds,
                    right_inds=right_inds,
                    eps=cutoff,
                    cutoff_mode='rel',
                    max_bond=max_bond,
                    inplace=True,
                    absorb=absorb,
                    ltags=left_tags,
                    rtags=right_tags,
                    keep_tags=False
                )
                
                left_val_tensor = self.getTensorFromTag(left_tags[0])
                left_tags_to_drop = list(filter(lambda tag: tag not in left_tags, left_val_tensor.tags))
                left_val_tensor.drop_tags(left_tags_to_drop)
                
                
                right_val_tensor =  self.getTensorFromTag(right_tags[0])
                right_tags_to_drop = list(filter(lambda tag: tag not in right_tags, right_val_tensor.tags))
                right_val_tensor.drop_tags(right_tags_to_drop)
               
                if verbose:
                    self.draw(f"Zipping up: {row}, {col}") 
                
               
            # Contract the row with the center of orthogonality  
            center_ortho_row = end_row
                
            # Contract our tensors and fetch the tensor plus its index tag
            center_ortho_contracted_tensor, center_ortho_contracted_tensor_tag = self.contract_tensors_in_range([center_ortho_row, center_ortho_row + 0.5], [col-col_radius, col+col_radius])
            
            tags = [TensorHelpers.getTag(center_ortho_row, col+1), 'T']
            
            # Drop all other tags and add our new tags
            center_ortho_contracted_tensor.drop_tags()
            center_ortho_contracted_tensor.add_tag(tags)
            
            if verbose:
                self.draw("Contracting top row tensor")
            
            # contract the final tensor
            last_row = end_row-1

            last_contracted_tensor, last_contracted_tensor_tag = self.contract_tensors_in_range([last_row, last_row + 0.5], [col-col_radius, col+col_radius])
            
            tags = [TensorHelpers.getTag(last_row, col+1), f'U[{last_row}]']
            
            last_contracted_tensor.drop_tags()
            last_contracted_tensor.add_tag(tags)
            
            # Be sure that the last contracted tensor shifts to the 
            self.shift_tensors_columns_in_range([0, last_row], [0, col], col, 1)
            if verbose: 
                self.draw("Zip up Finished")

            

            # Base case: return if we have a complte MPO
            if (len(self.tn.tensors) <= self.N):
                # Merge the tensors to the first column for better visualization
                retag_map = {}
                for t in self.tn.tensors:
                    t_old_tag = TensorHelpers.getRowColTagFromTensor(t)
                    t_row, t_col = TensorHelpers.getTensorRowCol(t)
                    retag_map[t_old_tag] = TensorHelpers.getTag(t_row, 0)
                    
                self.tn.retag(retag_map, inplace=True)
               
                if verbose: 
                    self.draw()
                
                # Get the tensors in the correct order
                sorted_tensors = self.get_sorted_tensors()
                
                # Reindex the index between these two tensors to have a specifc bond name
                reindex_map = {}
                sorted_tensors = self.get_sorted_tensors()
                for i in range(self.N - 1):
                    shared_inds = TensorHelpers.getSharedIndicesBetween(sorted_tensors[i], sorted_tensors[i+1])
                    reindex_map[shared_inds[0]] = f'z{i:03}'
                    
                self.tn.reindex(reindex_map, inplace=True)
                
                
                # Reindex the upper and lower indices of the tensors
                def get_lower_index(t):
                    # Sort through all indices and find the ones that are not bonds
                    inds = [Index(i) for i in t.inds if Index.isValid(i)]
                    return str(sorted(inds, key=lambda i: i.get_smallest_col())[-1])

                def get_upper_index(t):
                    inds = [Index(i) for i in t.inds if Index.isValid(i)]
                    return str(sorted(inds, key=lambda i: i.get_smallest_col())[0])

                
                reindex_map = {}
                for i in range(self.N):
                    t = sorted_tensors[i]
                    
                    upper_index = get_upper_index(t)
                    lower_index = get_lower_index(t)
                    
                    reindex_map[upper_index] = f'k{i:03}'
                    reindex_map[lower_index] = f'b{i:03}'
                    
                self.tn.reindex(reindex_map, inplace=True)
                    
                
                # Lastly, reorder all tensor indices to be in a correct and organized order
                for t in sorted_tensors:
                    sorted_inds = sorted(t.inds)
                    for i in range(len(sorted_inds)):
                        t.moveindex(sorted_inds[i], i, inplace=True)
                       
                if verbose: 
                    self.draw()
                return
            
            
            # Do zip down
            for row in range(end_row, self.N-1):
                cur_tensor, cur_tensor_tag = self.contract_tensors_in_range([row-0.5, row], [col-col_radius, col+col_radius])
                
                if verbose:
                    self.draw(f"Zip Down row {row}, col {col}")
                
                # Get the indices we're going to SVD on
                right_inds = [str(index) for index in self.getSVDIndices(cur_tensor, row, 'down')]
                left_inds = list(filter(lambda t: t not in right_inds,TensorHelpers.getTensorIndices(cur_tensor)))
                
                absorb = "right"
                right_tags = [f'VxS[{row}]', TensorHelpers.getTag(row+0.5, col+1), 'T']
                left_tags = [f'U[{row}]', TensorHelpers.getTag(row, col+1)]
                
                self.tn.replace_with_svd(
                    [cur_tensor_tag], 
                    left_inds=left_inds,
                    right_inds=right_inds,
                    eps=cutoff,
                    cutoff_mode='rel',
                    max_bond=max_bond,
                    inplace=True,
                    absorb=absorb,
                    ltags=left_tags,
                    rtags=right_tags,
                    keep_tags=False
                )
                
                left_val_tensor = self.getTensorFromTag(left_tags[0])
                left_tags_to_drop = list(filter(lambda tag: tag not in left_tags, left_val_tensor.tags))
                left_val_tensor.drop_tags(left_tags_to_drop)
                
                
                right_val_tensor =  self.getTensorFromTag(right_tags[0])
                right_tags_to_drop = list(filter(lambda tag: tag not in right_tags, right_val_tensor.tags))
                right_val_tensor.drop_tags(right_tags_to_drop)
                
                if verbose:
                    self.draw(f"Zip Down row {row}, col {col}")

            # Do last contraction operation
            last_row = self.N-1
            last_contracted_tensor, last_contracted_tensor_tag = self.contract_tensors_in_range([last_row-0.5, last_row], [col-col_radius, col+col_radius])
            
            last_tags = [TensorHelpers.getTag(last_row, col+1), 'T']
            last_contracted_tensor.drop_tags()
            last_contracted_tensor.add_tag(last_tags)
           
            if verbose: 
                self.draw("Zip Down Finished")
                
            # Lastly, shift the tags of our current tensors over by 2
            row_range = (0, self.N)
            col_range = (0, col+1)
            self.shift_tensors_columns_in_range(row_range, col_range, col, 2)
            
            if verbose: 
                self.draw("Shifted Over")
            
            # Increment our current column over by 2
            col += 2
            
            # Increment our ending row before starting a new column
            end_row += 1
            
    def shift_tensors_columns_in_range(self, row_range, col_range, target_col, shift_amount):
        '''
        Shifts the corresponding tensors in a given range over by a certain amount
        '''
        tensors_to_shift = self.getValidTensorsInRange(row_range, col_range)
        shift_retag_map = {}
        for t in tensors_to_shift:
            tensor_row, tensor_col = TensorHelpers.getTensorRowCol(t)
            tensor_tag = TensorHelpers.getRowColTagFromTensor(t)
            shift_retag_map[tensor_tag] = TensorHelpers.getTag(tensor_row, target_col+shift_amount)
        # Apply retagging 
        self.tn.retag(shift_retag_map, inplace=True)
        
    
    def draw(self, title=""):
        fix_dict = {}

        # # Layout the tensor network in a grid
        for i in np.arange(0, self.N*self.N, 0.5):
            for j in np.arange(0, self.N*self.N, 0.5):
                fix_dict[f'({i:.1f}, {j:.1f})'] = (j, -i)

        self.tn.draw(
            color=['P','H', 'C', 'T'], 
            figsize=(12, 12),
            margin=0.05,
            show_inds='bond-size',
            show_tags=True,
            initial_layout='shell',
            fix=fix_dict,
            font_size=10,
            title=title,
            edge_color='black',
            edge_alpha=1.0,
            edge_scale=1.0,
            arrow_linewidth=1.0,
            arrow_overhang=0.0,
            node_size=400,
            node_outline_darkness=0.0,
            node_outline_size=2.5
        )

    def print_tensors(self):
        '''
        Helper function to print out the tensors in the correct order
        '''
        sorted_tensors = self.get_sorted_tensors()
        for tensor in sorted_tensors:
            print(f"Tensor shape: {tensor.shape}")
            print(f"Tensor: {tensor}")
            print(f"Tensor Data:\n {tensor.data}", end='\n\n')