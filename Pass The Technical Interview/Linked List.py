class Node:
    def __init__(self, value, next_node=None):
        self.value = value
        self.next_node = next_node

    def get_value(self):
        return self.value

    def get_next_node(self):
        return self.next_node

    def set_next_node(self, next_node):
        self.next_node = next_node


class LinkedList:
    def __init__(self, value=None):
        self.head_node = Node(value)

    def get_head_node(self):
        return self.head_node

    def insert_beginning(self, new_value):
        new_node = Node(new_value)
        new_node.set_next_node(self.head_node)
        self.head_node = new_node

    def stringify_list(self):
        string_list = ""
        current_node = self.get_head_node()
        while current_node:
            if current_node.get_value() is not None:
                string_list += str(current_node.get_value()) + "\n"
            current_node = current_node.get_next_node()
        return string_list

    def remove_node(self, value_to_remove):
        current_node = self.get_head_node()
        if current_node.get_value() == value_to_remove:
            self.head_node = current_node.get_next_node()
        else:
            while current_node:
                next_node = current_node.get_next_node()
                if next_node.get_value() == value_to_remove:
                    current_node.set_next_node(next_node.get_next_node())
                    current_node = None
                else:
                    current_node = next_node


# TODO: add ability to append to end of linked list. Can be done by keeping track of head and tail nodes
# TODO: add the ability to remove a node via index in the linked list. This can be calculated in the method or kept
#  track of while new nodes are added (probably best to go with the former)
# TODO: How do you think you would remove all nodes that have a specific value? Try building a method to do that!


# Get nth last value in a linked list:
def nth_last_node(linked_list, n):
    current = None
    tail_seeker = linked_list.head_node
    count = 1
    while tail_seeker:
        tail_seeker = tail_seeker.get_next_node()
        count += 1
        if count >= n + 1:
            if current is None:
                current = linked_list.head_node
            else:
                current = current.get_next_node()
    return current


# pointers at different speeds (finds the middle node)
def find_middle(linked_list):
    fast_counter = linked_list.head_node
    slow_counter = linked_list.head_node

    while fast_counter is not None:
        fast_counter = fast_counter.get_next_node()
        if fast_counter is not None:
            fast_counter = fast_counter.get_next_node()
            slow_counter = slow_counter.get_next_node()
    return slow_counter
