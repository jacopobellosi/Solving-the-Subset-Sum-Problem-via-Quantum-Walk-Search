
import numpy as np
import sys
from math import comb
from qat.lang.AQASM import classarith # Force arithmetic library loading
from qat.lang.AQASM import Program
from qat.qpus import PyLinalg
from qat.lang.AQASM.gates import H, X
from qat.lang.AQASM.routines import QRoutine
from qatext.utils.qatmgmt.program import ProgramWrapper
from qatext.qroutines import qregs_init
from qatext.qroutines.datastructure.sliding_sort_array import insert_lw, insert_ld, delete
import cssp

def run_test(name, gate_fn, n, m, input_prep_fn=None):
    print(f"\n--- Testing {name} ---")
    prog = Program()
    # We need appropriate registers for the gate
    # For insert/delete: 1 val + n cells (total n*m + m qubits) + ancillas
    # Let's allocate manually to keep it simple.
    
    # Val register (m qubits)
    qr_val = prog.qalloc(m)
    # Array register (n * m qubits)
    qr_arr = prog.qalloc(n * m)
    
    # Initialize some state
    # Prepare val = 1 (binary 0...01)
    if m > 0:
        prog.apply(X, qr_val[0]) # LSB (or MSB depending on convention but non-zero)

    # Prepare array: sort of sorted?
    # Let's just put 0s. 0 is sorted.
    # Logic requires sorted input for insert.
    
    # Custom prep if needed
    if input_prep_fn:
        input_prep_fn(prog, qr_val, qr_arr)

    # Apply Gate
    gate = gate_fn(n, m)
    prog.apply(gate, qr_val, qr_arr)
    
    # Apply Inverse
    prog.apply(gate.dag(), qr_val, qr_arr)
    
    # Simulate
    circ = prog.to_circ()
    print(f"Circ: {circ.nbqbits} qubits, {len(circ.ops)} gates")
    sys.stdout.flush()
    
    qpu = PyLinalg()
    res = qpu.submit(circ.to_job())
    
    # Check if we returned to valid state (original state)
    # Original state: val=1, arr=0.
    # We check if any probability is non-zero for states where val!=1 or arr!=0.
    
    found_garbage = False
    for sample in res:
        if sample.probability < 1e-10: continue
        val = sample.state.lsb_int
        # Expected value:
        # val=1 at start? 
        # qr_val is first m qubits. qr_arr next n*m.
        # X on qr_val[0] -> state is |1> (val=1, arr=0).
        # Depending on endianness.
        
        # We just want to know if there is >1 state with prob > 0.
        # Since we start with a basis state (no superposition), we should end with ONLY that basis state.
        pass
        
    # Count non-zero prob states
    states = [s for s in res if s.probability > 1e-6]
    if len(states) != 1:
        print(f"FAIL: {len(states)} states found. Expected 1.")
        for s in states:
            print(f"  State {s.state.lsb_int}: {s.probability}")
        # return False # Don't stop, just fail
    else:
        print("PASS: Single state output (Reversible).")
        # return True

def debug_incremental():
    n = 3
    k = 1
    m = 2
    
    # Test insert_lw
    # insert_lw expects val and array.
    # We test insert_lw(n, m) operation followed by its inverse.
    run_test("insert_lw (G * G_dag)", insert_lw, n, m)
    
    # Test delete
    # Now that delete uses insert_lw.dag(),
    # delete(n, m) is G_dag.
    # We test G_dag * G.
    # This requires input to G_dag to be valid.
    
    def prep_valid_delete(prog, val, arr):
        # Set array element 0 to 1?
        # Assuming array is interpreted as little endian?
        # arr[0] is first m qubits.
        prog.apply(X, arr[0])
    
    run_test("delete (G_dag * G)", delete, n, m, prep_valid_delete)

if __name__ == "__main__":
    debug_incremental()

