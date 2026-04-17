"""
CSSP Grover Search - Matrix-based approach
============================================
Solves the Constrained Subset Sum Problem using Grover's algorithm
with a concrete unitary matrix for the VBE preparation.

This bypasses the myQLM linker bug by:
1. Compiling the Dicke+BIX preparation into a unitary matrix
2. Creating an AbstractGate from that matrix  
3. Using .dag() on the matrix gate (just conjugate transpose, no linker)

Usage: python cssp_grover.py True
"""
import sys
import os
import numpy as np
from math import comb

from qat.lang.AQASM import Program, classarith
from qat.lang.AQASM.gates import H, X, Z
from qat.lang.AQASM.routines import QRoutine
from qat.lang.AQASM.qftarith import QFT
from qat.qpus import PyLinalg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qatext.qroutines import bix
from qatext.qroutines import qregs_init as qregs
from qatext.qroutines.arith import cuccaro_arith
from qatext.qroutines.hamming_weight_generate.bartschiE19 import generate
from qatext.utils.qatmgmt.program import ProgramWrapper
from qatext.utils.qatmgmt.routines import QRoutineWrapper

QPU = PyLinalg()


def extract_unitary_matrix(n, k, m, sorted_values):
    """
    Extract the unitary matrix of the Dicke+BIX preparation circuit
    by simulating all basis states.
    
    The preparation acts on: n (dicke) + k*m (s_1) + (n-k)*m (s_0) qubits
    """
    n_qubits = n + k * m + (n - k) * m
    dim = 2 ** n_qubits
    print(f"Extracting {dim}x{dim} unitary matrix ({n_qubits} qubits)...")
    
    # Build the preparation circuit once to get the compiled version
    prep_prog = Program()
    qbits = prep_prog.qalloc(n_qubits)
    
    # We need to apply generate and BIX on the right qubit subsets
    # dicke: qbits[0:n]
    # s_1: qbits[n:n+k*m]  
    # s_0: qbits[n+k*m:n+k*m+(n-k)*m]
    
    # Create a QRoutine for the preparation
    prep_qrout = QRoutine()
    dicke_wires = prep_qrout.new_wires(n)
    s1_wires = [prep_qrout.new_wires(m) for _ in range(k)]
    s0_wires = [prep_qrout.new_wires(m) for _ in range(n - k)]
    
    prep_qrout.apply(generate(n, k), dicke_wires)
    prep_qrout.apply(bix.bix_matrix_compile_time(n, 1, m, k, sorted_values),
                     dicke_wires, *s1_wires, *s0_wires)
    
    prep_prog.apply(prep_qrout, qbits)
    
    # Compile - this works because there's no .dag() involved
    prep_circ = prep_prog.to_circ(link=[classarith, cuccaro_arith])
    
    # Extract unitary by simulating each basis state
    unitary = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        # Create a job that initializes to basis state |i> and measures all qubits
        # Use the statevector approach: simulate |i> through the circuit
        job = prep_circ.to_job(job_type="SAMPLE", nbshots=0)
        # Alternative: use the circuit's matrix directly if available
        pass
    
    # Actually, use PyLinalg's ability to get the full statevector
    # by submitting a job with amplitude extraction
    for col in range(dim):
        # Build circuit: initialize to |col>, apply prep
        p = Program()
        q = p.qalloc(n_qubits)
        
        # Set qubits to binary representation of col
        for bit in range(n_qubits):
            if (col >> bit) & 1:
                p.apply(X, q[bit])
        
        # Apply the preparation QRoutine
        p.apply(prep_qrout, q)
        
        c = p.to_circ(link=[classarith, cuccaro_arith])
        
        # Get full statevector by measuring all qubits in SAMPLE mode
        # with amplitudes
        job = c.to_job()
        result = QPU.submit(job)
        
        for sample in result:
            # Convert state to integer index
            state_int = sample.state.int
            unitary[state_int, col] = np.sqrt(sample.probability) * np.exp(
                1j * sample.amplitude_phase if hasattr(sample, 'amplitude_phase') else 0)
    
    # Actually, let me use a simpler approach with the observable-free simulation
    # PyLinalg can return the full amplitude vector
    
    return unitary


def extract_unitary_simple(n, k, m, sorted_values):
    """
    Simpler extraction: build circuit for each basis state input,
    get the output statevector using PyLinalg's amplitude interface.
    """
    n_qubits = n + k * m + (n - k) * m
    dim = 2 ** n_qubits
    print(f"Extracting {dim}x{dim} unitary ({n_qubits} qubits)...")
    
    # Build prep QRoutine once
    prep_qrout = QRoutine()
    dicke_wires = prep_qrout.new_wires(n)
    s1_wires = [prep_qrout.new_wires(m) for _ in range(k)]
    s0_wires = [prep_qrout.new_wires(m) for _ in range(n - k)]
    prep_qrout.apply(generate(n, k), dicke_wires)
    prep_qrout.apply(bix.bix_matrix_compile_time(n, 1, m, k, sorted_values),
                     dicke_wires, *s1_wires, *s0_wires)
    
    unitary = np.zeros((dim, dim), dtype=complex)
    
    for col in range(dim):
        p = Program()
        q = p.qalloc(n_qubits)
        
        # Initialize to |col>
        for bit in range(n_qubits):
            if (col >> bit) & 1:
                p.apply(X, q[bit])
        
        p.apply(prep_qrout, q)
        c = p.to_circ(link=[classarith, cuccaro_arith])
        
        # Get ALL amplitudes by requesting all qubits
        job = c.to_job(qubits=list(range(n_qubits)))
        result = QPU.submit(job)
        
        for sample in result:
            row = sample.state.int
            # PyLinalg gives probability and state; we need amplitude
            # For real circuits, amplitude = sqrt(prob) * phase
            # Use the sample.amplitude if available
            if hasattr(sample, 'amplitude'):
                unitary[row, col] = sample.amplitude
            else:
                # Fallback: can only get probability, not phase
                unitary[row, col] = np.sqrt(sample.probability)
        
        if col % 100 == 0 and col > 0:
            print(f"  Column {col}/{dim} done")
    
    print(f"  Matrix extracted. Unitarity check: ||UU†-I|| = {np.linalg.norm(unitary @ unitary.conj().T - np.eye(dim)):.6e}")
    return unitary


def oracle(n, k, m, n_qubits_sum, target_value):
    """Oracle: same as cssp.py but as a standalone function."""
    qrw = QRoutineWrapper(QRoutine())
    node_s_ones = qrw.qarray_wires(k, m, "s_1", int)
    sum_reg = qrw.qarray_wires(1, n_qubits_sum, "sum", int)
    qrout_sum = classarith.add(n_qubits_sum, m)
    with qrw.compute():
        for j in range(k):
            qrw.apply(qrout_sum, sum_reg, node_s_ones[j])
        qrw.apply(
            qregs.initialize_qureg_to_complement_of_int(
                target_value, n_qubits_sum, False), sum_reg)
    qrw.apply(Z.ctrl(n_qubits_sum - 1), sum_reg)
    qrw.uncompute()
    return qrw


def main(n, k, values, target_sum, to_simulate=False):
    m = max(values).bit_length()
    n_qubits_sum = int(np.ceil(np.log2(k))) + m
    sorted_values = sorted(values)
    n_prep_qubits = n + k * m + (n - k) * m
    
    print(f"n={n}, k={k}, m={m}, values={values}, target={target_sum}")
    print(f"Preparation qubits: {n_prep_qubits}")
    
    # Step 1: Extract the preparation unitary matrix
    prep_matrix = extract_unitary_simple(n, k, m, sorted_values)
    
    # Step 2: Create AbstractGate with matrix_generator so simulator can use it
    from qat.lang.AQASM.misc import AbstractGate
    
    n_prep_q = int(np.log2(prep_matrix.shape[0]))
    
    # Forward gate
    PREP_GATE = AbstractGate("U_PREP", [np.ndarray],
                             matrix_generator=lambda mat: mat,
                             arity=lambda mat: int(np.log2(mat.shape[0])))
    prep_gate = PREP_GATE(prep_matrix)
    
    # Inverse gate (separate name to avoid any linker confusion)
    PREP_DAG_GATE = AbstractGate("U_PREP_DAG", [np.ndarray],
                                 matrix_generator=lambda mat: mat,
                                 arity=lambda mat: int(np.log2(mat.shape[0])))
    prep_gate_dag = PREP_DAG_GATE(prep_matrix.conj().T)
    
    # Step 3: Build the Grover circuit
    prw = ProgramWrapper(Program())
    dicke = prw.qarray_alloc(n, 1, "dicke", str)
    node_s_ones = prw.qarray_alloc(k, m, "s_1", int)
    node_s_zeros = prw.qarray_alloc(n - k, m, "s_0", int)
    sum_reg = prw.qarray_alloc(1, n_qubits_sum, "sum", int)
    
    # Collect all prep qubits in order: dicke, s_1, s_0
    prep_qubits = list(dicke) + list(node_s_ones) + list(node_s_zeros)
    
    # Initial preparation
    prw.apply(prep_gate, prep_qubits)
    
    # Grover loop
    n_iters = int(np.ceil(np.sqrt(comb(n, k))))
    print(f"Grover iterations: {n_iters}")
    
    for _ in range(n_iters):
        # Oracle
        qf_ora = oracle(n, k, m, n_qubits_sum, target_sum)
        prw.apply(qf_ora, node_s_ones, sum_reg)
        
        # Diffuser: U_prep * Z_0 * U_prep^dag
        prw.apply(prep_gate_dag, prep_qubits)  # U_prep^dag
        
        for j in range(n):
            prw.apply(X, dicke[j])
        prw.apply(Z.ctrl(n - 1), dicke)
        for j in range(n):
            prw.apply(X, dicke[j])
        
        prw.apply(prep_gate, prep_qubits)  # U_prep
    
    print("Program qubits")
    for k_name, v in prw._qregnames_to_properties.items():
        print(k_name, v.slic)
    
    # Compile - prep_gate is a concrete matrix, no linker issues
    cr = prw.to_circ(link=[classarith, cuccaro_arith])
    print(cr.statistics())
    
    job = cr.to_job(qubits=[*node_s_ones])
    
    if to_simulate:
        res = QPU.submit(job)
        for sample in res:
            print(sample.probability, sample.state)


if __name__ == '__main__':
    to_simulate = len(sys.argv) > 1 and sys.argv[1].lower() == 'true'
    print(f"To simulate is {to_simulate}")
    
    values = [1, 2, 3]
    n = len(values)
    k = 1
    ts = 3
    
    main(n, k, values, ts, to_simulate=to_simulate)
