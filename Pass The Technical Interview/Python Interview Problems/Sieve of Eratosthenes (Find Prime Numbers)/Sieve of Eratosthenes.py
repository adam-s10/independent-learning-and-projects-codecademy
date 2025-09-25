import math


def sieve_of_eratosthenes(limit):
    true_indices = []

    # handle edge cases
    if limit <= 1:
        return true_indices

    # create the output list
    output = [True] * (limit + 1)

    # mark 0 and 1 as non-prime
    output[0] = False
    output[1] = False

    # iterate up to the square root of the limit
    for i in range(2, limit + 1):
        if output[i]:
            j = i * 2
            # mark all multiples of i as non-prime
            while j <= limit:
                output[j] = False
                j += i

    # remove non-prime numbers
    output_with_indices = list(enumerate(output))
    true_indices = [index for (index, value) in output_with_indices if value]
    return true_indices


primes = sieve_of_eratosthenes(13)
print(primes)  # should return [2, 3, 5, 7, 11, 13]


def optimized_sieve_of_eratosthenes(limit):
    # handle edge cases
    if limit <= 1:
        return []

    # create the output list
    output = [True] * (limit + 1)

    # mark 0 and 1 as non-prime
    output[0] = False
    output[1] = False

    # iterate up to the square root of the limit
    for i in range(2, math.floor(math.sqrt(limit))):
        if output[i]:
            j = i ** 2  # initialize j to square of i

            # mark all multiples of i as non-prime
            while j <= limit:
                output[j] = False
                j += i

    # remove non-prime numbers
    output_with_indices = list(enumerate(output))
    trues = [index for (index, value) in output_with_indices if value]
    return trues


# primes = optimized_sieve_of_eratosthenes(20)
# print(primes)  # returns [2, 3, 5, 7, 11, 13, 17, 19]
