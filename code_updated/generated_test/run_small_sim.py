import sys
import os

# Ensure the current directory is in the path to import cssp
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cssp import main

def run_simulation():
    # Define a small problem instance
    # Set X = {1, 2, 3}
    values = [1, 2, 3]
    n = len(values)
    
    # We want to find a subset of size k=2 that sums to 3
    # The only solution is {1, 2}
    k = 2
    target_sum = 3
    
    print(f"--- Starting Small Simulation ---")
    print(f"Set: {values}")
    print(f"Subset size k: {k}")
    print(f"Target sum: {target_sum}")
    print(f"Expected solution: {{1, 2}}")
    print(f"-------------------------------")

    # Run the main function with to_simulate=True
    # low_width=True uses the optimized insertion circuit
    main(n, k, values, target_sum, low_width=True, to_simulate=True)

if __name__ == "__main__":
    run_simulation()
