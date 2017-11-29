"""
Library with utility functions for mlhack
"""

from collections import Iterable
import itertools as it
import csv

def recursive_map(itera, item_func=lambda v: v, itera_func=lambda v: v):
    """
    Appliest function to every function in nested iterators

    Arguments:
    itera (Iterable) - Iterable to apply
    item_func (lambda) - function to apply to every leaf item
    itera_func (lampda) - function to apply to every Iterable
    """
    if isinstance(itera, str):
        return item_func(itera)

    try:
        return itera_func(map(nested_to_list, itera))
    except TypeError:
        return item_func(itera)


def nested_to_list(itera):
    """
    Converts nested iterable to nested list

    Arguments:
    itera - Iterable to convert
    """

    return recursive_map(itera, itera_func=list)

def flatten_level(itera, level):
    """
    Flatterns nested iterators level

    Arguemnts:
    itera (Iterator) - iterator
    level (int) - level (root = 0) to flatten
    """

    if not isinstance(itera, Iterable) or isinstance(itera, str):
        return itera
    if level <= 0:
        return it.chain.from_iterable(itera)
    else:
        return map(lambda v: flatten_level(v, level-1), itera)


def flatten_levels(itera, levels):
    """
    Flatterns nested iterators levels

    Arguemnts:
    itera (Iterator) - iterator
    level (int) - levels (root = 0) to flatten
    """

    result = itera
    for level in sorted(levels, reverse=True):
        result = flatten_level(result, level)
    return result

def nested_partial_print(itera, top=10):
    """
    Prints top items of nested iteration

    Arguemnts:
    itera (Iterator) - iterator
    top (int) - amount of items to print
    """

    for item in it.islice(itera, top):
        print(nested_to_list(item))

def read_csv_to_dict(path, key_selector, value_selector, encoding="utf8", **csv_kwargs):
    """
    Read CSV to doctionary

    Arguments:
    path - path to CSV file
    key_selector (lambda) - lambda function to select key (e.g. lambda row: row['key'])
    value_selector (lambda) - lambda function to select value (e.g. lambda row: row['key'])
    encoding (str) - encoding to use while reading file (default: utf8)
    **kwargs (named arguments) - arguemnts passed to csv.DictReader (see csv.DictReader for more info)
    """

    with open(path, encoding=encoding) as f:
        reader = csv.DictReader(f, **csv_kwargs)
        return {key_selector(row):value_selector(row) for row in reader}

