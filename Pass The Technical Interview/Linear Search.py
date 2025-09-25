# A list of the ingredients for tuna sushi
recipe = ["nori", "tuna", "soy sauce", "sushi rice"]
target_ingredient = "avocado"


# Only searches until first occurrence of target_value is found
def linear_search(search_list, target_value):
    for idx in range(len(search_list)):
        if search_list[idx] == target_value:
            return idx
    raise ValueError("{0} not in list".format(target_value))


print(linear_search(recipe, target_ingredient))

# Search list
test_scores = [88, 93, 75, 100, 80, 67, 71, 92, 90, 83]


# Find max using linear search
def linear_search_find_max(search_list):
    maximum_score_index = None
    for idx in range(len(search_list)):
        print(search_list[idx])
        if maximum_score_index is None or search_list[idx] > search_list[maximum_score_index]:
            maximum_score_index = idx
    return maximum_score_index


# Function call
highest_score = linear_search_find_max(test_scores)

# Prints out the highest score in the list
print(highest_score)

# Search list and target value
tour_locations = ["New York City", "Los Angeles", "Bangkok", "Istanbul", "London", "New York City", "Toronto"]
target_city = "New York City"


# Linear Search Algorithm for finding duplicates
def linear_search_duplicates(search_list, target_value):
    matches = []
    for idx in range(len(search_list)):
        if search_list[idx] == target_value:
            matches.append(idx)
    if len(matches) != 0:
        return matches
    else:
        raise ValueError("{0} not in list".format(target_value))


# Function call
tour_stops = linear_search(tour_locations, target_city)
print(tour_stops)
