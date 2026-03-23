import sys
import os
import numpy as np

# Ensure the current directory is in the path to import cssp
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cssp import main
from qat.qpus import PyLinalg

def run_test():
    values = [1, 2, 3]
    n = 3
    k = 1
    target = 3
    
    # We will just patch PyLinalg in cssp to let us capture the results
    print("Testing cssp.py without cleanup...")
    import cssp
    cssp.main(n, k, values, target, low_width=True, to_simulate=True)
    
if __name__ == "__main__":
    run_test()
