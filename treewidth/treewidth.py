#!/usr/bin/env python3
from  scipy import *
from  pylab import *
import sys
import numpy as np
import random as rd
import time

class treewidth():
    
    def __init__(self, file):
        td_file = open(file + ".td", 'r', encoding="utf-8")
        gr_file = open(file + ".gr", 'r', encoding="utf-8")
        
        line = td_file.readline()
        while line[0] == 'c':
            line = td_file.readline()
        s, td, N, w, n = line.split() #N:nb bags
        self.N = int(N)
        self.w = int(w)
        
        line = gr_file.readline()
        while line[0] == 'c':
            line = gr_file.readline()

        p, tw, n, m = line.split()
        self.n = int(n)
        self.m = int(m)

        self.graph_edges = [[] for _ in range(self.n + 1)]

        for x in range(0, self.m):
            u, v = gr_file.readline().split()

            self.graph_edges[int(u)].append(int(v))
            self.graph_edges[int(v)].append(int(u))

        self.bags = [[] for _ in range(self.N + 1)]

        for x in range(1, self.N + 1):
            line = td_file.readline().split()[2:]
            for y in line:
                self.bags[x].append(int(y))

        self.tree_edges = [[] for _ in range(self.N + 1)]

        line = td_file.readline()
        while line != "":
            u, v = line.split()

            self.tree_edges[int(u)].append(int(v))
            self.tree_edges[int(v)].append(int(u))

            line = td_file.readline()
        
        self.visited_nodes = [0] * (self.N + 1)
        self.tree = [[] for _ in range(self.N + 1)]

    def __call__(self):
        self.visited_nodes = [0] * (self.N + 1)
        self.tree = [[] for _ in range(self.N + 1)]
        self.build_tree(1)

        root = self.find_optimal_subsets(1)
        best = max(root.values(), default=0)
        print(f"n = {self.n}, w = {self.w - 1}, best: {best}")

    def is_independent(self, subset):
        """Checks if the given subset of vertices is independent."""
        if len(subset) <= 1:
            return True

        subset_set = set(subset)
        for vertex in subset:
            neighbors = self.graph_edges[vertex]
            for neighbor in neighbors:
                if neighbor in subset_set and neighbor != vertex:
                    return False
        return True


    def build_tree(self, x):
        """DFS to direct the given tree"""
        self.visited_nodes[x] = 1
        neighbours = self.tree_edges[x] #retrieve nodes neighbouring x
        for n in neighbours:
            if self.visited_nodes[n] == 0:
                self.tree[x].append(n)
                self.build_tree(n)

    def to_bitstring(self, v_t):
        """Generates all possible subsets of a given set of vertices and checks their independence"""
        n = 2**len(v_t) - 1
        t_u = {}
        for x in range(n + 1):
            b = format(x, f'0{len(v_t)}b')  # x to binary str with zeros to len of v_t.
            u = [v_t[i] for i in range(len(v_t)) if b[i] == '1']  # create subset u based on bitstr b
            if self.is_independent(u) != -1:
                t_u[b] = self.is_independent(u)  # store u & value if indep
        return t_u
    
    def solve(self, parent_subsets, child_subsets, parent_node, child_node):
        """Combines results from parent and child nodes in the decomposition to compute the best value for each subset"""
        parent_bag = self.bags[parent_node]
        child_bag = self.bags[child_node]
        for parent_subset in parent_subsets:
            parent_vertices = [parent_bag[i] for i, bit in enumerate(parent_subset) if bit == '1']
            intersection_parent_child = set(parent_vertices) & set(child_bag)

            best_value = 0
            for child_subset in child_subsets:
                child_vertices = [child_bag[i] for i, bit in enumerate(child_subset) if bit == '1']
                if set(child_vertices) & set(parent_bag) == intersection_parent_child:
                    value = child_subsets[child_subset] - len(set(parent_vertices) & set(child_vertices))
                    best_value = max(best_value, value)
            
            parent_subsets[parent_subset] += best_value
        return parent_subsets

    def find_optimal_subsets(self, node):
        """
        Recursively finds the optimal subsets for the given node and its descendants in the tree
        by computing bitstrings for the node's bag and combining these with optimal subsets from its children.
        """
        # Check if the node is a leaf (has no children)
        if not self.tree[node]:
            # Build and return all possible subsets (bitstrings) for the node's bag if it has no children
            return self.to_bitstring(self.bags[node])

        # For non-leaf nodes, process each child
        node_subsets = self.to_bitstring(self.bags[node])  # All subsets for the current node
        for child in self.tree[node]:
            child_subsets = self.find_optimal_subsets(child)
            node_subsets = self.combine_subsets(node_subsets, child_subsets, node, child)  # Combine current and child subsets
        return node_subsets

    def combine_subsets(self, parent_subsets, child_subsets, parent_node, child_node):
        """
        Combines the subsets of a parent node with those of a child node by considering dependencies
        between their bags and updating parent subsets based on optimal child subsets.
        """
        return self.solve(parent_subsets, child_subsets, parent_node, child_node)

if __name__ == "__main__":
    start_time = time.time()
    file = f"./data/{sys.argv[1]}"
    tw = treewidth(file)
    tw()
    print("%s seconds" % (time.time() - start_time))
