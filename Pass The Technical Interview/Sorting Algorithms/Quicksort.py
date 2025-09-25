from random import randrange, shuffle


def quicksort(my_list, start, end):
    # this portion of list has been sorted
    if start >= end:
        return
    print("Running quicksort on {0}".format(my_list[start: end + 1]))
    # select random element to be pivot
    pivot_idx = randrange(start, end + 1)
    pivot_element = my_list[pivot_idx]
    print("Selected pivot {0}".format(pivot_element))
    # swap random element with last element in sub-lists
    my_list[end], my_list[pivot_idx] = my_list[pivot_idx], my_list[end]

    # tracks all elements which should be to left (lesser than) pivot
    less_than_pointer = start

    for i in range(start, end):
        # we found an element out of place
        if my_list[i] < pivot_element:
            # swap element to the right-most portion of lesser elements
            print("Swapping {0} with {1}".format(my_list[i], my_list[less_than_pointer]))
            my_list[i], my_list[less_than_pointer] = my_list[less_than_pointer], my_list[i]
            # tally that we have one more lesser element
            less_than_pointer += 1
    # move pivot element to the right-most portion of lesser elements
    my_list[end], my_list[less_than_pointer] = my_list[less_than_pointer], my_list[end]
    print("{0} successfully partitioned".format(my_list[start: end + 1]))
    # recursively sort left and right sub-lists
    quicksort(my_list, start, less_than_pointer - 1)
    quicksort(my_list, less_than_pointer + 1, end)


unsorted_list = [5, 3, 1, 7, 4, 6, 2, 8]
shuffle(unsorted_list)
print("PRE SORT: ", unsorted_list)
print(quicksort(unsorted_list, 0, len(unsorted_list) - 1))
print("POST SORT: ", unsorted_list)
