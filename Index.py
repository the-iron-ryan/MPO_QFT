import re

'''
Basic Index Class
'''
class Index:
    def __init__(self, *args, **kwargs):

        # When given a single argument, assume its a connected tag
        if len(args) == 1:
            self.nodes = sorted(self._parse_tag(args[0]))
        elif len(args) == 2:
            self.nodes = sorted([args[0], args[1]])

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