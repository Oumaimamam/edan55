#!/usr/bin/env python3
"""the simple randomized algorithm for the maxcut problem"""
import sys
import random
import matplotlib.pyplot as plt
import numpy as np

def data_import(filepath):
    """imports the data from the file"""
    with open(filepath, 'r', encoding='utf-8') as file:
        _, _ = file.readline().split()
        edges = [tuple(line.split()) for line in file]
    return edges

def build_random_set(vertices):
    """returns a random set of vertices"""
    return {v for v in vertices if random.randint(0, 1) == 1}

def get_weight(edge_dict, v1, v2):
    """Returns the weight of the edge between v1 and v2"""
    return int(edge_dict.get((v1, v2), 0)) # Return 0 if the edge does not exist

def calculate_initial_cut_weight(edges, A):
    """Calculates the initial weight of the cut based on set A."""
    cut_weight = 0
    for v1, v2, weight in edges:
        if (v1 in A and v2 not in A) or (v1 not in A and v2 in A):
            cut_weight += int(weight)
    return cut_weight

def randomized_max_cut(edges):
    """Randomized maximum cut of the graph"""
    vertices = set()
    edge_dict = {}
    for v1, v2, weight in edges:
        vertices.add(v1)
        vertices.add(v2)
        edge_dict[(v1, v2)] = weight
        edge_dict[(v2, v1)] = weight

    rnd_set = build_random_set(vertices)
    remaining_set = vertices - rnd_set
    cut = 0
    for v1 in rnd_set:
        for v2 in remaining_set:
            cut += get_weight(edge_dict, v1, v2)
    return cut

def greedy_swapping_max_cut(edges):
    """greedy swapping max cut algorithm"""
    vertices = {v for edge in edges for v in edge[:2]}
    A = set()
    max_cut_weight = calculate_initial_cut_weight(edges, A)  # Initial cut weight

    improvement = True
    while improvement:
        improvement = False
        for v in vertices:
            new_A = A.symmetric_difference({v})  # Create a new set that is A with v swapped
            new_cut_weight = calculate_initial_cut_weight(edges, new_A)
            if new_cut_weight > max_cut_weight:
                max_cut_weight = new_cut_weight
                A = new_A
                improvement = True
                break  # Found an improvement, break and start looking for the next improvement
    return max_cut_weight

def randomized_swapping_max_cut(edges):
    """randomized swapping max cut algorithm"""
    vertices = {v for edge in edges for v in edge[:2]} # Extract vertices from edges
    A = set(random.sample(vertices, k=len(vertices) // 2))  # Start with a random partition

    improved = True
    while improved:
        improved = False
        best_increase = 0
        vertex_to_swap = None

        for v in vertices:
            A_new = A.symmetric_difference({v})
            new_cut_weight = calculate_initial_cut_weight(edges, A_new)
            current_cut_weight = calculate_initial_cut_weight(edges, A)

            if new_cut_weight > current_cut_weight and new_cut_weight - current_cut_weight > best_increase:
                best_increase = new_cut_weight - current_cut_weight
                vertex_to_swap = v
                improved = True

        if improved and vertex_to_swap:
            A.symmetric_difference_update({vertex_to_swap})  # Swap the vertex
    return calculate_initial_cut_weight(edges, A)

def run_experiments(algorithm, edges, runs=100):
    """Runs the algorithm multiple times and returns the average and max cut size"""
    cut_sizes = [algorithm(edges) for _ in range(runs)]
    avg_cut_size = np.mean(cut_sizes)
    max_cut_size = max(cut_sizes)
    return avg_cut_size, max_cut_size, cut_sizes

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print('Usage: ./algo_r.py "<data/file.txt>"')
        sys.exit(1)

    # Import the data
    data_filepath = sys.argv[1]
    edges = data_import(data_filepath)

    # Run the algorithms
    max_cut_r = randomized_max_cut(edges)
    print(f"* The max_cut weight for random algo = {max_cut_r}")


    max_cut_s = greedy_swapping_max_cut(edges)
    print(f"* The max_cut weight for swapping algo = {max_cut_s}")

    max_cut_rs = randomized_swapping_max_cut(edges)
    print(f"* The max_cut weight for random swapping algo = {max_cut_rs}")

    # Run experiments for performance evaluation
    avg_cut_size_r, max_cut_size_r, cut_sizes_r = run_experiments(randomized_max_cut, edges)
    avg_cut_size_s, max_cut_size_s, cut_sizes_s = run_experiments(greedy_swapping_max_cut, edges)
    avg_cut_size_rs, max_cut_size_rs, cut_sizes_rs = run_experiments(randomized_swapping_max_cut, edges)

    print(f"* Randomized max cut: avg={avg_cut_size_r}, max={max_cut_size_r}")
    print(f"* Greedy swapping max cut: avg={avg_cut_size_s}, max={max_cut_size_s}")
    print(f"* Randomized swapping max cut: avg={avg_cut_size_rs}, max={max_cut_size_rs}")

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.hist(cut_sizes_r, bins=20, alpha=0.75)
    plt.title('Cut Sizes Distribution for Algorithm R')
    plt.xlabel('Cut Size')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(cut_sizes_s, bins=20, alpha=0.75)
    plt.title('Cut Sizes Distribution for Algorithm S')
    plt.xlabel('Cut Size')

    plt.subplot(1, 3, 3)
    plt.hist(cut_sizes_rs, bins=20, alpha=0.75)
    plt.title('Cut Sizes Distribution for Algorithm RS')
    plt.xlabel('Cut Size')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
