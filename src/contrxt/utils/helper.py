import os
from typing import List


def union(
    lst_1: List,
    lst_2: List
)-> int:
    """Return total number of features from two lists.

    Args:
        lst_1 (List): _description_
        lst_2 (List): _description_

    Returns:
        int: _description_
    """
    return len(set(lst_1).union(set(lst_2)))

def jaccard_distance(
    lst_1: List,
    lst_2: List,
)->int:
    """Returns the Jaccard distance between two lists.

    Args:
        lst_1 (List): List to compare
        lst_2 (List): List to compare

    Returns:
        int: Jaccard distance between the two lists.
    """
    set_1 , set_2 = set(lst_1), set(lst_2)
    n_union = len(set_1.union(set_2))
    n_intersection = len(set_1.intersection(set_2))
    jaccard_idx = n_intersection / n_union

    return 1 - jaccard_idx

if __name__ == "__main__":
    pass