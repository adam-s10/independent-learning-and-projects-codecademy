from LinkedList import LinkedList

my_list = LinkedList()

my_list.insert_beginning("Node 1")
my_list.insert_beginning("Node 2")
my_list.insert_beginning("Node 3")
my_list.insert_beginning("Node 4")

# Add code here:
found_node = my_list.find_node_recursively("Node 5", my_list.head_node)
print(found_node)
