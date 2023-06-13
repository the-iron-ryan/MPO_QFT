import re
import ast
from typing import Any

'''
Basic Index Class
'''
class Index:
    def __init__(self, *args, **kwargs):

        # When given a single argument, assume its a connected tag
        if len(args) == 1:
            parsed_str_values = self._parse_tag(args[0])
            self.nodes = sorted([ast.literal_eval(s_val) for s_val in parsed_str_values])
        elif len(args) == 2:
            self.nodes = sorted([ast.literal_eval(args[0]), ast.literal_eval(args[1])])

    def _parse_tag(self, tag: str):
        # Match between parenthesis
        return re.findall(r'\(.+?\)', tag)

    def contains(self, tag: str):
        return tag in self.nodes

    def getIntersections(self, other):
        return list(filter(lambda tag: tag in other.nodes, self.nodes))
    
    def getConnectedTag(self, tag):
        '''
        Helper method to get the other connected tag in this index, if it exists.
        If it doesn't exist, return None
        '''
        if self.contains(tag):
            return self.nodes[0] if tag == self.nodes[1] else self.nodes[1]
        else:
            return None
            
    def replace(self, old_tag: str, new_tag):
        '''
        Replaces the old_tag with the new_tag. Returns True if successful, False otherwise
        '''
        if self.contains(old_tag):
            self.nodes.remove(old_tag)
            self.nodes.append(new_tag)
            self.nodes = sorted(self.nodes)
            return True
        else:
            return False 

    def __str__(self) -> str:
        return f'{self.nodes[0]}<->{self.nodes[1]}'
    
    def get_smallest_row(self):
        return min(self.nodes, key=lambda node: node[0])
    def get_largest_row(self):
        return max(self.nodes, key=lambda node: node[0])
    def get_smallest_col(self):
        return min(self.nodes, key=lambda node: node[1])
    def get_largest_col(self):
        return max(self.nodes, key=lambda node: node[1])
   
    @classmethod 
    def isValid(cls, tag: str):
        '''
        Helper class method to check if a tag is valid
        '''
        return len(re.findall(r'\(.+?\)', tag)) == 2