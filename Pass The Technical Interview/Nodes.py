# Nodes typically hold data and a pointer to another node. This pointer can be to a parent or
# child node, to both, to multiple children, or multiple children and parent nodes.
#
# Nodes might need to be updated during the execution of our code and links need to be rewritten. We must be careful
# as we can 'orphan' a node we might need in the future
class Node:
    def __init__(self, value, link_node=None):
        self.value = value
        self.link_node = link_node

    def set_link_node(self, link_node):
        self.link_node = link_node

    def get_link_node(self):
        return self.link_node

    def get_value(self):
        return self.value


# Add your code below:
yacko = Node('likes to yak')
wacko = Node('has a penchant for hoarding snacks')
dot = Node('enjoys spending time in movie lots')

# yacko -> dot -> wacko
yacko.set_link_node(dot)
dot.set_link_node(wacko)

# get data inside of dot via link with yacko
dots_data = yacko.get_link_node().get_value()
# get data inside of wacko via link with dot
wackos_data = dot.get_link_node().get_value()
# print values returned
print(dots_data)
print(wackos_data)
