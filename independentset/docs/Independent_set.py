#!/usr/bin/env python3
"""several algorithms to find the size of the biggest indepedant set in our graph"""
import sys
import numpy as np
import matplotlib.pyplot as plt

def neighbours(u, graph):
    """Returns the indices of the neighbors of the node idx_u."""
    return np.flatnonzero(graph[u])

def indep_set_r0(graph, n, nb_calls=0):
    """Recursively solves for the maximum independent set size."""
    nb_calls += 1
    if n==0:
        return 0, nb_calls

    new_n, new_graph = n, graph.copy()
    isolated_nodes = [i for i in range(n) if np.all(new_graph[i] == 0)]

    if isolated_nodes:
        new_graph = np.delete(new_graph, isolated_nodes, axis=0)
        new_graph = np.delete(new_graph, isolated_nodes, axis=1)
        new_n -= len(isolated_nodes)
        result, nb_calls = indep_set_r0(new_graph, new_n, nb_calls)
        return 1 + result, nb_calls

    degrees = new_graph.sum(axis=1)
    u = np.argmax(degrees)
    u_neighbours = np.flatnonzero(new_graph[u])

    reduced_graph_excl_neighbors = np.delete(new_graph, u_neighbours, axis=0)
    reduced_graph_excl_neighbors = np.delete(reduced_graph_excl_neighbors, u_neighbours, axis=1)
    option1, nb_calls = indep_set_r0(reduced_graph_excl_neighbors, new_n - len(u_neighbours), nb_calls)

    reduced_graph_excl_u = np.delete(new_graph, u, axis=0)
    reduced_graph_excl_u = np.delete(reduced_graph_excl_u, u, axis=1)
    option2, nb_calls = indep_set_r0(reduced_graph_excl_u, new_n - 1, nb_calls)

    return max(1 + option1, option2), nb_calls


def indep_set_r1(graph, n, nb_calls=0):
    """Recursively solves for the maximum independent set size."""
    nb_calls += 1
    if n == 0:
        return 0, nb_calls

    new_graph = graph.copy()

    terminal_nodes = [i for i in range(n) if np.count_nonzero(new_graph[i]) == 1]
    if terminal_nodes:
        for u in terminal_nodes:
            neighbor = np.flatnonzero(new_graph[u])[0]
            new_graph = np.delete(new_graph, [u, neighbor], axis=0)
            new_graph = np.delete(new_graph, [u, neighbor], axis=1)
            new_n = n - 2
            result, nb_calls = indep_set_r1(new_graph, new_n, nb_calls)
            return 1 + result, nb_calls

    isolated_nodes = [i for i in range(n) if np.all(new_graph[i] == 0)]
    if isolated_nodes:
        new_graph = np.delete(new_graph, isolated_nodes, axis=0)
        new_graph = np.delete(new_graph, isolated_nodes, axis=1)
        new_n = n - len(isolated_nodes)
        result, nb_calls = indep_set_r1(new_graph, new_n, nb_calls)
        return len(isolated_nodes) + result, nb_calls

    degrees = new_graph.sum(axis=1)
    u = np.argmax(degrees)
    u_neighbours = np.flatnonzero(new_graph[u])
    reduced_graph_excl_neighbors = np.delete(new_graph, u_neighbours, axis=0)
    reduced_graph_excl_neighbors = np.delete(reduced_graph_excl_neighbors, u_neighbours, axis=1)
    option1, nb_calls = indep_set_r1(reduced_graph_excl_neighbors, n - len(u_neighbours), nb_calls)

    reduced_graph_excl_u = np.delete(new_graph, u, axis=0)
    reduced_graph_excl_u = np.delete(reduced_graph_excl_u, u, axis=1)
    option2, nb_calls = indep_set_r1(reduced_graph_excl_u, n - 1, nb_calls)

    return max(1 + option1, option2), nb_calls


def indep_set_r2(graph, n, nb_calls=0):
    """Recursively solves for the maximum independent set size."""
    nb_calls += 1
    if n == 0:
        return 0, nb_calls

    new_graph = graph.copy()

    two_neighbours_nodes = [i for i in range(n) if np.count_nonzero(new_graph[i]) == 2]
    if two_neighbours_nodes:
        for u in two_neighbours_nodes:
            neighbors = np.flatnonzero(new_graph[u])
            new_graph = np.delete(new_graph, [u, neighbors[0], neighbors[1]], axis=0)
            new_graph = np.delete(new_graph, [u, neighbors[0], neighbors[1]], axis=1)
            new_n = n - 3
            result, nb_calls = indep_set_r2(new_graph, new_n, nb_calls)
            return 1 + result, nb_calls

    terminal_nodes = [i for i in range(n) if np.count_nonzero(new_graph[i]) == 1]
    if terminal_nodes:
        for u in terminal_nodes:
            neighbor = np.flatnonzero(new_graph[u])[0]
            new_graph = np.delete(new_graph, [u, neighbor], axis=0)
            new_graph = np.delete(new_graph, [u, neighbor], axis=1)
            new_n = n - 2
            result, nb_calls = indep_set_r2(new_graph, new_n, nb_calls)
            return 1 + result, nb_calls

    isolated_nodes = [i for i in range(n) if np.all(new_graph[i] == 0)]
    if isolated_nodes:
        new_graph = np.delete(new_graph, isolated_nodes, axis=0)
        new_graph = np.delete(new_graph, isolated_nodes, axis=1)
        new_n = n - len(isolated_nodes)
        result, nb_calls = indep_set_r2(new_graph, new_n, nb_calls)
        return len(isolated_nodes) + result, nb_calls

    degrees = new_graph.sum(axis=1)
    u = np.argmax(degrees)
    u_neighbours = np.flatnonzero(new_graph[u])
    reduced_graph_excl_neighbors = np.delete(new_graph, u_neighbours, axis=0)
    reduced_graph_excl_neighbors = np.delete(reduced_graph_excl_neighbors, u_neighbours, axis=1)
    option1, nb_calls = indep_set_r2(reduced_graph_excl_neighbors, n - len(u_neighbours), nb_calls)

    reduced_graph_excl_u = np.delete(new_graph, u, axis=0)
    reduced_graph_excl_u = np.delete(reduced_graph_excl_u, u, axis=1)
    option2, nb_calls = indep_set_r2(reduced_graph_excl_u, n - 1, nb_calls)

    return max(1 + option1, option2), nb_calls


def main():
    """main function"""
    file = open(f"{sys.argv[1]}", "r", encoding="utf-8")
    data = [list(map(int,line.strip().split())) for line in file]
    n, graph = data.pop(0)[0], np.array(data)[0:]
    print("the result of algorithm R0: ", indep_set_r0(graph, n))
    print("the result of algorithm R1: ", indep_set_r1(graph, n))
    print("the result of algorithm R2: ", indep_set_r2(graph, n))
    ####plots####
    # graph_sizes = range(30, 80, 10)
    # call_counts_r0 = []
    # call_counts_r1 = []
    # call_counts_r2 = []
    # for size in graph_sizes:
    #     filename = f"../data/g{size}.in"
    #     with open(filename, 'r', encoding="utf8") as file:
    #         data = [list(map(int, line.strip().split())) for line in file]
    #         n, graph = data.pop(0)[0], np.array(data)

    #         _, calls_r0 = indep_set_r0(graph, n)
    #         _, calls_r1 = indep_set_r1(graph, n)
    #         _, calls_r2 = indep_set_r2(graph, n)

    #         call_counts_r0.append(calls_r0)
    #         call_counts_r1.append(calls_r1)
    #         call_counts_r2.append(calls_r2)
    # plt.figure(figsize=(10, 6))
    # plt.plot(graph_sizes, np.log(call_counts_r0), label='R0', marker='o')
    # plt.plot(graph_sizes, np.log(call_counts_r1), label='R1', marker='o')
    # plt.plot(graph_sizes, np.log(call_counts_r2), label='R2', marker='o')
    # plt.xlabel('Graph Size (n)')
    # plt.ylabel('Log of Number of Recursive Calls')
    # plt.title('Recursive Call Complexity')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("recursive_calls_comparison.png")
    # plt.show()

if __name__== '__main__':
    main()