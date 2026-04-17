import time
import random
import itertools
import matplotlib.pyplot as plt
import math

def subset_sum_brute_force(values, k, target_sum):
    """
    Solves the Constrained Subset Sum Problem using brute force.
    Checks all possible combinations of size 'k' from 'values'.
    Returns the first valid combination and the number of attempts,
    or None if no such subset exists.
    """
    attempts = 0
    for subset in itertools.combinations(values, k):
        attempts += 1
        if sum(subset) == target_sum:
            return subset, attempts
    return None, attempts

def generate_random_ssp_instance(n, m_bits):
    """
    Generates a random instance of the SSP.
    Returns the values array, a valid target_sum (to ensure a solution exists in the worst case),
    and the chosen k size (usually n/2 to maximize combinations).
    """
    max_val = (1 << m_bits) - 1
    values = [random.randint(1, max_val) for _ in range(n)]
    
    # Choose k to be half of n to maximize the binomial coefficient C(n, k) (worst case scenario)
    k = max(1, n // 2)
    
    # Pick a random valid subset to guarantee a solution exists
    solution_subset = random.sample(values, k)
    target_sum = sum(solution_subset)
    
    return values, k, target_sum

def run_benchmark_and_plot(max_n=22, m_bits=10):
    """
    Runs the brute force algorithm for increasing sizes of 'n',
    records the time taken, and plots the resulting graph.
    """
    print(f"Starting Classical Brute Force Benchmark (Max n={max_n}, Element size={m_bits} bits)")
    
    n_values = []
    times = []
    
    for n in range(4, max_n + 1, 2):  # Step by 2 to see the curve cleanly
        values, k, target_sum = generate_random_ssp_instance(n, m_bits)
        
        print(f"Testing n={n}, k={k}... ", end="", flush=True)
        
        start_time = time.perf_counter()
        solution, attempts = subset_sum_brute_force(values, k, target_sum)
        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        
        n_values.append(n)
        times.append(elapsed_time)
        
        print(f"Found {solution} in {elapsed_time:.4f}s ({attempts} checks)")
        
        # Safety break if it takes more than 10 seconds to compute a single instance,
        # to avoid freezing the user's PC on huge numbers.
        if elapsed_time > 10.0:
            print("Time limit exceeded (>10s). Stopping benchmark early to prevent freezing.")
            break

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, times, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    
    plt.title('Brute Force Subset-Sum Execution Time vs Set Size (N)', fontsize=14)
    plt.xlabel('Size of Initial Set N (with k = N/2)', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.yscale('log') # Use logarithmic scale because brute force is exponential!
    plt.ylabel('Execution Time (seconds) - Log Scale', fontsize=12)
    
    plt.tight_layout()
    
    print("\nBenchmark complete! Displaying graph...")
    plt.show()

if __name__ == "__main__":
    # Run the benchmark up to N=26 (adjust as needed, N=26 takes a few seconds)
    run_benchmark_and_plot(max_n=52, m_bits=12)
