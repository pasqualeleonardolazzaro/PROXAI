import random

def keep_random_element_in_place(lst):
    if not lst or len(lst) == 0:
        return []  # Return None if the list is empty
    random_element = random.choice(lst)
    lst.clear()
    lst.append(random_element)
    return random_element