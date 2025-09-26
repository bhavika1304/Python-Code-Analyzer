import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
from dataflow_analyzer import DataFlowAnalyzer

code = """
def binary_search(arr, x):
    l, r = 0, len(arr)-1
    while l <= r:
        mid = (l+r)//2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            l = mid+1
        else:
            r = mid-1
    return -1

def factorial(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result
"""

tree = ast.parse(code)
for fn in tree.body:
    if isinstance(fn, ast.FunctionDef):
        analyzer = DataFlowAnalyzer()
        result = analyzer.analyze(fn)
        print(f"\n=== {fn.name} ===")
        print("Symbols:", result["symbols"])
        print("Loops:", result["loops"])
        print("Updates:", result["updates"])
