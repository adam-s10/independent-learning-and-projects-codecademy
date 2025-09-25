# Imagine a very heavy rainstorm over a highway that has many potholes and cracks. The rainwater will collect in the
# empty spaces in the road, creating puddles. Each puddle can only be as high as the road around it, as any excess water
# will just flow away.
#
# The capturing rainwater problem asks you to calculate how much rainwater would be trapped in the empty spaces in a
# histogram (a chart which consists of a series of bars) This can be represented in Python as an array filled with the
# values [4, 2, 1, 3, 0, 1, 2]. Imagine that rainwater has fallen over the histogram and collected between the bars.

# Like with the road, the amount of water that can be captured at any given space cannot be higher than the bounds
# around it. To solve the problem, we need to write a function that will take in an array of integers and calculate the
# total water captured. Our function would return 6 for the histogram above. There are multiple ways to solve this
# problem, but we are going to focus on a naive implementation and an optimized implementation.


def naive_solution(heights):
    total_water = 0
    for i in range(1, len(heights) - 1):
        left_bound = 0
        right_bound = 0
        # We only want to look at the elements to the left of i, which are the elements at the lower indices
        for j in range(i + 1):
            left_bound = max(left_bound, heights[j])

        # Likewise, we only want the elements to the right of i, which are the elements at the higher indices
        for j in range(i, len(heights)):
            right_bound = max(right_bound, heights[j])

        total_water += min(left_bound, right_bound) - heights[i]

    return total_water


def efficient_solution(heights):
    total_water = 0
    left_pointer = 0
    right_pointer = len(heights) - 1
    left_bound = 0
    right_bound = 0

    # Write your code here
    while left_pointer < right_pointer:
        if heights[left_pointer] <= heights[right_pointer]:
            left_bound = max(left_bound, heights[left_pointer])
            total_water += left_bound - heights[left_pointer]
            left_pointer += 1
        else:
            right_bound = max(right_bound, heights[right_pointer])
            total_water += right_bound - heights[right_pointer]
            right_pointer -= 1
    return total_water


test_array = [4, 2, 1, 3, 0, 1, 2]
print(efficient_solution(test_array))
print(naive_solution(test_array))
# Print 6
