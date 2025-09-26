import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
import networkx as nx
from cfg_builder import CFGBuilder

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
"""

tree = ast.parse(code)
func_node = tree.body[0]

builder = CFGBuilder()
cfg = builder.build(func_node)

print("Nodes:")
for n, d in cfg.nodes(data=True):
    print(n, d)

print("\nEdges:")
for u, v in cfg.edges():
    print(f"{u} -> {v}")
import matplotlib.pyplot as plt
pos = nx.spring_layout(cfg)
labels = nx.get_node_attributes(cfg, "label")
nx.draw(cfg, pos, with_labels=True, labels=labels, node_size=2500, node_color="lightblue", font_size=8, font_weight="bold", arrows=True)
plt.show()
