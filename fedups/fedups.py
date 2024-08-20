import numpy as np
import random as rd
import sys
import time

class Road:
    
    def __init__(self, to, time, prob):
        self.to = int(to)
        self.time = float(time)
        self.prob = float(prob)

class Fedups:
    
    def __init__(self, file):
        with open(file, 'r') as f:
            first_line = f.readline().strip().split()
            
            self.N = int(first_line[0])  # number of intersections
            self.M = int(first_line[1])  # number of roads
            self.H = int(first_line[2])  # home node
            self.F = int(first_line[3])  # FedUPS starting node
            self.P = int(first_line[4])  # PostNHL starting node
            
            self.pMatrix = np.zeros((self.N, self.N))
            self.time = np.zeros(self.N)
            
            self.links = [[] for _ in range(self.N)]
            
            for _ in range(self.M):
                line = f.readline().strip().split()
                u = int(line[0])
                v = int(line[1])
                t = float(line[2])
                puv = float(line[3])
                pvu = float(line[4])
                self.pMatrix[u, v] = puv
                self.pMatrix[v, u] = pvu
                self.time[u] += t * puv
                self.time[v] += t * pvu
                if puv > 0:
                    self.links[u].append(v)
                if pvu > 0:
                    self.links[v].append(u)

    def monte_carlo(self, start, dest, num_runs):
        total_time = 0.0
        for _ in range(num_runs):
            current = start
            time_spent = 0
            while current != dest:
                rand = rd.random()
                cumulative_prob = 0.0
                for road in self.links[current]:
                    cumulative_prob += self.pMatrix[current, road]
                    if cumulative_prob >= rand:
                        time_spent += self.time[current]
                        current = road
                        break
            total_time += time_spent
        return total_time / num_runs
    
    def is_connected(self, start, dest):
        visited = [False] * self.N
        self._dfs(start, visited)
        return visited[dest]

    def _dfs(self, node, visited):
        visited[node] = True
        for neighbor in self.links[node]:
            if not visited[neighbor]:
                self._dfs(neighbor, visited)
    
    def markov(self, start, dest):
        if not self.is_connected(start, dest):
            print(f"No path from {start} to {dest}. Returning infinity.")
            return float('inf')
        
        A = np.copy(self.pMatrix)
        b = -1 * np.copy(self.time)
        
        N = len(A)
        for i in range(N):
            A[i, i] -= 1
        
        solution = np.linalg.solve(A, b)
        return solution[start]
    
    def monte_carlo_converge_3_digits(self, start, dest, num_runs, markov_result, tolerance=1e-3):
        for _ in range(10):
            current_result = self.monte_carlo(start, dest, num_runs)
            relative_error = abs(current_result - markov_result) / abs(markov_result)
            if relative_error > tolerance:
                return False
        return True
    
    def run_convergence_test(self, file):
        num_runs = 250000
        tolerance = 1e-3
        max_runs = 1000000

        start_time = time.time()
        fedups_markov = self.markov(self.F, self.H)
        postnhl_markov = self.markov(self.P, self.H)

        while num_runs <= max_runs:
            fedups_converged = self.monte_carlo_converge_3_digits(self.F, self.H, num_runs, fedups_markov, tolerance)
            postnhl_converged = self.monte_carlo_converge_3_digits(self.P, self.H, num_runs, postnhl_markov, tolerance)

            if fedups_converged and postnhl_converged:
                duration = time.time() - start_time
                print(f"Monte Carlo with {num_runs} runs has converged in {duration} seconds.")
                return num_runs, duration
            else:
                num_runs *= 2

        print(f"Convergence was not achieved in {max_runs} runs.")
        return None, None

    def run_Monte_Carlo(self):
        #Run Monte Carlo Simulation
        num_runs = 10000
        start_time = time.time()
        fedups_monte = self.monte_carlo(self.F, self.H, num_runs)
        postnhl_monte = self.monte_carlo(self.P, self.H, num_runs)
        monte_duration = time.time() - start_time
        print(f"Monte Carlo: FedUPS: {fedups_monte}, PostNHL: {postnhl_monte}, Duration: {monte_duration} seconds")

    def run_Markov(self):
        # Run Markov Chain Algorithm
        start_time = time.time()
        fedups_markov = self.markov(self.F, self.H)
        postnhl_markov = self.markov(self.P, self.H)
        markov_duration = time.time() - start_time
        print(f"Markov: FedUPS: {fedups_markov}, PostNHL: {postnhl_markov}, Duration: {markov_duration} seconds")


if __name__ == "__main__":
    file = f'./data/{sys.argv[1]}'
    fedups = Fedups(file)
    #fedups.run_Monte_Carlo()
    fedups.run_Markov()
    #fedups.run_convergence_test(file)
