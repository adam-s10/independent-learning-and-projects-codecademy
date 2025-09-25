# Define sum_to_one() below...
def sum_to_one(n):
    if n == 1:
        return n
    print('Recursing with input: {0}'.format(n))
    return n + sum_to_one(n - 1)


'''
A function that  takes an integer as an input and returns the sum of all numbers from the input down to 1 recursively

Base Case: n is 1
Recursive Step: n + sum_to_one(n - 1)'''
# uncomment when you're ready to test
print(sum_to_one(7))


def sum_to_one_iterative(n):
    result = 0
    for num in range(n, 0, -1):
        result += num
    return result


'''
A function that  takes an integer as an input and returns the sum of all numbers from the input down 
to 1 iteratively
'''
sum_to_one_iterative(4)


# num is set to 4, 3, 2, and 1
# 10


# Define factorial() below:
def factorial(n):
    if n < 2:
        return 1
    return n * factorial(n - 1)


'''
Base Case: n < 2 OR n ≤ 1
Recursive Step: n * factorial(n - 1)
'''
print(factorial(12))


def power_set(my_list):
    # base case: an empty list
    if len(my_list) == 0:
        return [[]]
    # recursive step: subsets without first element
    power_set_without_first = power_set(my_list[1:])
    # subsets with first element
    with_first = [[my_list[0]] + rest for rest in power_set_without_first]
    # return combination of the two
    return with_first + power_set_without_first


'''A power set is a list of all subsets of the values in a list. Producing subsets requires a runtime of at least 
O(2^N), we’ll never do better than that because a set of N elements creates a power set of 2^N elements'''
universities = ['MIT', 'UCLA', 'Stanford', 'NYU']
power_set_of_universities = power_set(universities)

for power in power_set_of_universities:
    print(power)


# define flatten() below...
def flatten(my_list):
    result = []
    for l in my_list:
        if isinstance(l, list):
            print('List found!')
            flat_list = flatten(l)
            result += flat_list
        else:
            result.append(l)
    return result


'''
Base Case: the element is not a list
Recursive Step: add list that was inside the list to result
'''
# reserve for testing...
planets = ['mercury', 'venus', ['earth'], 'mars', [['jupiter', 'saturn']], 'uranus', ['neptune', 'pluto']]
print(flatten(planets))


# define the fibonacci() function below...
def fibonacci(n):
    if n == 1 or n == 0:
        return n
    print('n is', n)
    return fibonacci(n - 2) + fibonacci(n - 1)


'''
Base Case: n is 1 or n is 0
Recursive Step: fibonacci(n - 2) + fibonacci(n - 1)
Runtime: O(2^n)
'''
print(fibonacci(10))
# set the appropriate runtime:
# 1, logN, N, N^2, 2^N, N!
fibonacci_runtime = "2^N"


# Define build_bst() below... Binary search tree
def build_bst(my_list):
    if len(my_list) == 0:
        return 'No Child'

    middle_idx = len(my_list) // 2
    middle_value = my_list[middle_idx]
    print('Middle index:', middle_idx)
    print('Middle value:', middle_value)

    tree_node = {'data': middle_value}
    tree_node.update({'left_child': build_bst(my_list[:middle_idx])})
    tree_node.update({'right_child': build_bst(my_list[middle_idx + 1:])})
    return tree_node


'''
Base Case: No Child is found OR list length = 0
Recursive Step: The input must be divided into 2 halves, find middle index of list, store value at middle index, 
                make tree node with a "data" value as middle index value, 
                assign tree node’s "left child" to a recursive call using the left half of the list,
                assign tree node’s "right child" to a recursive call using the right half of the list, 
                return the tree node
Runtime: N*log(n)
Runtime Reasoning: - N is the length of our input list.
                   - Our tree will be logN levels deep, meaning there will logN times where a new parent-child 
                     relationship is created.
                   - If we have an 8 element list, the tree is 3 levels deep: 2**3 == 8.
                   - Each recursive call is going to copy approximately N elements when the left and right halves of the 
                     list are passed to the recursive calls. We’re reducing by 1 each time (the middle_value), but 
                     that’s a constant factor.
                   - Putting that together, we have N elements being copied logN levels for a big O of N*logN.
'''
# For testing
sorted_list = [12, 13, 14, 15, 16]
binary_search_tree = build_bst(sorted_list)
print(binary_search_tree)

# fill in the runtime as a string
# 1, logN, N, N*logN, N^2, 2^N, N!
runtime = "N*logN"

print(sorted(['a', 'm', 'z', 'g']))
