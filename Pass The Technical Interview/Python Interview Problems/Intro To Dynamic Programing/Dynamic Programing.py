memo = {}


def fibonacci(num):
    answer = None
    # Write your code here
    if num in memo:
        answer = memo[num]
    elif num == 0 or num == 1:
        answer = num
    else:
        answer = fibonacci(num - 1) + fibonacci(num - 2)
        memo[num] = answer
    return answer


'''Create a dictionary that contains the results of certain fibonacci numbers as to not constantly recalculate them 
through execution as input increases'''


# Test your code with calls here:
print(fibonacci(20))
print(fibonacci(200))
