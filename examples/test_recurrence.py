import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
from recurrence_extractor import RecurrenceExtractor

code = """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)

def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

def binary_search(arr, l, r, x):
    if r >= l:
        mid = (l+r)//2
        if arr[mid] == x:
            return mid
        elif arr[mid] > x:
            return binary_search(arr, l, mid-1, x)
        else:
            return binary_search(arr, mid+1, r, x)
    return -1

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr)//2
        L = arr[:mid]
        R = arr[mid:]
        merge_sort(L)
        merge_sort(R)
        return L + R
    return arr
"""

tree = ast.parse(code)
for fn in tree.body:
    if isinstance(fn, ast.FunctionDef):
        extractor = RecurrenceExtractor(fn.name).analyze(fn)
        print(f"\n=== {fn.name} ===")
        print("Recursive calls:", extractor.recursive_calls)
        print("Recurrence:", extractor.recurrence)
        print("Complexity:", extractor.complexity)
