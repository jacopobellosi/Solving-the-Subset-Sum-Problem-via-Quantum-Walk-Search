# Import math utilities (e.g., combinatorics)
from math import comb

import numpy as np
# Import myQLM libraries to build quantum circuits (AQASM)
from qat.lang.AQASM import classarith
from qat.lang.AQASM.gates import H, X, Z
from qat.lang.AQASM.program import Program
from qat.lang.AQASM.qftarith import QFT
from qat.lang.AQASM.routines import QRoutine
from qat.qpus import PyLinalg

# Import project submodules (the specific building blocks from the paper)
from qatext.qroutines import bix
from qatext.qroutines import qregs_init
from qatext.qroutines import qregs_init as qregs
from qatext.qroutines.arith import cuccaro_arith
# These implement the "History Independent" sorted array data structure (Sec. III-A)
from qatext.qroutines.datastructure.sliding_sort_array import (  
    insert_ld, insert_lw, delete) # ld = Low-Depth, lw = Low-Width
from qatext.qroutines.hamming_weight_generate.bartschiE19 import generate
from qatext.utils.qatmgmt.program import ProgramWrapper
from qatext.utils.qatmgmt.routines import QRoutineWrapper

# Algebraic quantum simulator
QPU = PyLinalg()


# =====================================================================
# U_U OPERATOR (Update Operator) - Described in "Sec. III-D. Operator UU"
# Generates a uniform superposition of all Johnson graph vertices 
# adjacent to the current one (differing by exactly 1 element).
# =====================================================================
def update(n, k, m, insert, delete_fn):
    qrw = QRoutineWrapper(QRoutine())

    # =====================================================================
    # ALLOCATION: The `qarray_wires` method acts as a memory allocator for MyQLM. 
    # It reserves a contiguous block of qubits to form a 2D array-like quantum register.
    # The parameters are: (number_of_elements, width_in_qubits_per_element, unique_name, data_type).
    # =====================================================================
    # Paper Table I: node_s_ones is the register S containing the k elements of the currently visited subset
    node_s_ones = qrw.qarray_wires(k, m, "s_1", int)
    # Paper Table I: node_s_zeros is the register S' containing the remaining n-k elements (the complement subset)
    node_s_zeros = qrw.qarray_wires(n - k, m, "s_0", int)
    # Paper Table I: node_t_ones is the register T, an empty register where we will build the candidate adjacent vertex
    node_t_ones = qrw.qarray_wires(k, m, "t_1", int)
    # Paper Table I: node_t_zeros is the register T', the complement of the candidate adjacent vertex
    node_t_zeros = qrw.qarray_wires(n - k, m, "t_0", int)
    # Paper Table I: alpha_ones is the register \alpha, where we will temporarily store the element extracted from S
    alpha_ones = qrw.qarray_wires(1, m, "a_1", int) 
    # Paper Table I: alpha_zeros is the register \alpha', where we will temporarily store the element extracted from S'
    alpha_zeros = qrw.qarray_wires(1, m, "a_0", int) 
    # Paper Table I: wstate_ones is the register \omega, used to host the Dicke State |Wk> for sampling from S
    wstate_ones = qrw.qarray_wires(k, 1, "w_1", str)
    # Paper Table I: wstate_zeros is the register \omega', used to host the Dicke State |W(n-k)> for sampling from S'
    wstate_zeros = qrw.qarray_wires(n - k, 1, "w_0", str)
    # Initialize the Insertion (INS) and Deletion (DEL) circuits.
    # The exact implementation (Low-Width vs Low-Depth) is passed dynamically as arguments 'insert' and 'delete_fn'.
    qrout_insert_ones = insert(k, m)
    qrout_insert_zeros = insert(n - k, m)
    qrout_delete_ones = delete_fn(k, m)
    qrout_delete_zeros = delete_fn(n - k, m)

    # =====================================================================
    # 1. Copy S into T
    # =====================================================================
    # Paper Step: "As a first step, we then copy the contents of S into T, and similarly, we copy S' into T'."
    # This is done using an array of CNOT gates where S (node_s_ones) acts as control and T (node_t_ones) as target.
    qrw.apply(qregs_init.copy_array_of_registers(k, m), node_s_ones,
              node_t_ones)
    qrw.apply(qregs_init.copy_array_of_registers(n - k, m), node_s_zeros,
              node_t_zeros)

    # =====================================================================
    # 2. Uniform sample from T and T'
    # =====================================================================
    # Paper Step: "To do so, we generate the Dicke states |D1_k> and |D1_(n-k)> on \omega and \omega', respectively"
    # These states are superpositions where only a single qubit is |1>, meaning they act as perfectly uniform random selectors.
    qrw.apply(generate(k, 1), wstate_ones)
    qrw.apply(generate(n - k, 1), wstate_zeros)
    
    # Paper Step: "The unique qubit of \omega having value |1> is used as a control qubit to sample a unique element from T and copy it to \alpha"
    # This is the "C-COPY" procedure: a loop of controlled copies. Only the element whose corresponding \omega bit is |1> will be physically written into \alpha.
    for j in range(k):
        qrw.apply(
            qregs_init.copy_register(m).ctrl(), wstate_ones[j], node_s_ones[j],
            alpha_ones)
            
    # We repeat the exact same C-COPY sampling procedure for the complement set, grabbing one random element from S' into \alpha'.
    for j in range(n - k):
        qrw.apply(
            qregs_init.copy_register(m).ctrl(), wstate_zeros[j],
            node_s_zeros[j], alpha_zeros)
            
    # =====================================================================
    # 3. Insertion and deletion (Building the adjacent vertex)
    # =====================================================================
    # Paper Step: "the deletion circuit (DEL) is the conjugate transpose of the insertion circuit INS"
    # Here we delete the element we just sampled (stored in \alpha) from the main subset S. 
    # This leaves a "blank space |⊥" at the end of S.
    qrw.apply(qrout_delete_ones, alpha_ones, node_s_ones)
    
    # We also delete the sampled complement element (in \alpha') from the complement subset S'.
    qrw.apply(qrout_delete_zeros, alpha_zeros, node_s_zeros)

    # Paper Step: "Both implementations ensure that at the end of an insertion/deletion procedure, the resulting set is ordered."
    # Now we cross-insert! The element removed from S' (\alpha') is inserted into S to fill its blank space.
    # The element removed from S (\alpha) is inserted into S'. 
    # This swap of a single item defines an "adjacent vertex" on the Johnson Graph.
    # Apply the insertion circuit; sliding_sort_array.py provides both 
    # Low-Width and Low-Depth implementations (insert_lw and insert_ld).
    qrw.apply(qrout_insert_ones, alpha_zeros, node_s_ones)
    qrw.apply(qrout_insert_zeros, alpha_ones, node_s_zeros)
    return qrw


# =====================================================================
# U_f OPERATOR (Oracle Operator) - Described in "Sec. III-C. Operator Uf"
# The paper states: U_f acts as a phase flip for correctly "marked" subsets.
# It computes the sum of elements, compares it with target p, and applies Z.
# =====================================================================
def oracle(n, k, m, n_qubits_sum, target_value):
    qrw = QRoutineWrapper(QRoutine())  # Prepare AQASM subroutine wrapper
    # Retrieve our candidate subset S, which holds the k elements (each m qubits long)
    node_s_ones = qrw.qarray_wires(k, m, "s_1", int)
    # Paper Step: "additional register denoted by sum of size m [n_qubits_sum here]"
    # This register starts at 0 and will accumulate the total of S
    sum_reg = qrw.qarray_wires(1, n_qubits_sum, "sum", int)
    # Construct the mathematical addition circuit (citing Cuccaro et al. [25])
    qrout_sum = classarith.add(n_qubits_sum, m)
    # Using 'compute()' block so we can easily uncompute it later (quantum reversibility rule)
    with qrw.compute():
        # Paper Step: "We sequentially apply k quantum adders to compute the sum"
        for j in range(k):
            # Mathematically: sum = sum + S[j]
            qrw.apply(qrout_sum, sum_reg, node_s_ones[j])
        # Paper Step: "Then we initialize a register of size m with the target value p"
        # However, to save qubits, we compute the target value p bitwise into 'sum_reg' using XOR.
        # This acts as an equality check: if sum == p, the output state of sum_reg becomes all 1s.
        qrw.apply(
            qregs.initialize_qureg_to_complement_of_int(
                target_value, n_qubits_sum, False), sum_reg)
    # Paper Step: "An m-controlled Z gate is executed on a single qubit 
    #              previously initialized in the |-_> state."
    # Here, we use a multi-controlled Z gate (Z.ctrl) triggered on all bits of sum_reg.
    # It flips the phase ONLY if the sum exactly matched p (meaning sum_reg became all 1s).
    qrw.apply(Z.ctrl(n_qubits_sum - 1), sum_reg)
    
    # Paper Step: "The adders are then run in reverse to clear the sum register."
    # Uncompute everything (reverses the XOR and the k adders) restoring sum_reg to 0. -->the sign of the oracle is flipped if the sum matches the target
    qrw.uncompute()
    return qrw


# =====================================================================
# MAIN - Global construction of the MNRS Quantum Walk Search
# Assembles all parts of the paper to solve the CSSP (Subset Sum Problem)
# =====================================================================
def main(n,
         k,
         values: list[int],
         target_sum: int,
         low_width=True,
         to_simulate=False):
         
    # If low-width is True, "insert_lw" is used (minimizes ancillary qubits). 
    # Otherwise "insert_ld" is used (minimizes circuit depth but uses more qubits). Sec III-D "Insertion and deletion"
    insert = insert_lw if low_width else insert_ld
    
    # m = number of bits required to encode the maximum value in the set (Sec III Introduction)
    m = max(values).bit_length()
    
    # delta = Theoretical spectral gap of the Johnson Graph (Sec II-D, page 169)
    delta = n / (k * (n - k))
    
    # len_s = Number of Quantum Phase Estimation (QPE) steps based on O(1/\sqrt{\delta})
    len_s = int(np.ceil(np.log2(np.pi / (2 * np.sqrt(delta)))))
    
    # n_qubits_sum = Number of qubits needed to safely hold the sum of k integers without overflow
    n_qubits_sum = int(np.ceil(np.log2(k))) + m

    sorted_values = sorted(values)
    prw = ProgramWrapper(Program())
    
    # === ALLOCATION OF REGISTERS DESCRIBED IN TABLE I ===
    # dicke = \sigma Register 
    dicke = prw.qarray_alloc(n, 1, "dicke", str)
    # node_s_ones = S Register
    node_s_ones = prw.qarray_alloc(k, m, "s_1", int)
    # node_s_zeros = S' Register
    node_s_zeros = prw.qarray_alloc(n - k, m, "s_0", int)
    # node_t_ones = T Register
    node_t_ones = prw.qarray_alloc(k, m, "t_1", int)
    # node_t_zeros = T' Register
    node_t_zeros = prw.qarray_alloc(n - k, m, "t_0", int)
    # alpha_ones = \alpha Register
    alpha_ones = prw.qarray_alloc(1, m, "a_1", int)
    # alpha_zeros = \alpha' Register
    alpha_zeros = prw.qarray_alloc(1, m, "a_0", int)
    # wstate_ones = \omega Register
    wstate_ones = prw.qarray_alloc(k, 1, "w_1", str)
    # wstate_zeros = \omega' Register
    wstate_zeros = prw.qarray_alloc(n - k, 1, "w_0", str)

    # qpe_s = Support register used for the Approximate Reflection U_ref(E)
    qpe_s = prw.qarray_alloc(len_s, 1, "qpe_s", str)
    # sum_reg = Support "sum" register for the CSSP target sums
    sum_reg = prw.qarray_alloc(1, n_qubits_sum, "sum", int)
    
    # Initialize the QPE support with Hadamard gates (creates a 50/50 superposition on logical channels)
    for qb in qpe_s:
        prw.apply(H, qb)

    # =====================================================================
    # INPUT PREPARATION OPERATOR (U_v) - Section "Input preparation U_v" (Sec II-C)
    # =====================================================================
    
    # Paper Step 1: "Dicke state". Generate the |D^n_k> state.
    # This is a uniform superposition of all n-length bitstrings with 
    # Hamming weight equal to k. Stored on register \sigma (here called 'dicke').
    prw.apply(generate(n, k), dicke)
    
    # Paper Step 2: "Vertex Binary Encoding (VBE)".
    # Reads the \sigma sequence (dicke state) as control indices.
    # Stores the elements of subset S into k distinct cells in increasing order
    # within the 'node_s_ones' register, and the complement subset S' into 'node_s_zeros'.
    # This invokes bix_matrix_compile_time, which internally handles the C-xi and C-SHIFT gates.
    prw.apply(bix.bix_matrix_compile_time(n, 1, m, k, sorted_values), dicke,
              node_s_ones, node_s_zeros)
              
    # 3) Execute the Update Operator (U_U) a first time to generate the adjacent paths in T and T' (Edge Superposition)
    qrw_update = update(n, k, m, insert, delete)
    prw.apply(qrw_update, node_s_ones, node_s_zeros, node_t_ones, node_t_zeros,
              alpha_ones, alpha_zeros, wstate_ones, wstate_zeros)

    # =====================================================================
    # MAIN MNRS AMPLITUDE AMPLIFICATION ALGORITHMIC LOOP
    # Repeats the Grover + Szegedy architecture O(1 / sqrt(epsilon)) times
    # =====================================================================
    # n_external_iters is the parameter O(1/\sqrt{\epsilon}) calculated via binomial n over k
    n_external_iters = int(np.ceil(np.sqrt(comb(n, k))))
    for _ in range(n_external_iters):
    
        # 1. ORACLE U_ref^\perp(v*) - Calls the phase inversion function on marked (solution) vertices
        qf_ora = oracle(n, k, m, n_qubits_sum, target_sum)  # Instantiate the Oracle operator U_f with target p
        prw.apply(qf_ora, node_s_ones, sum_reg)  # Apply the Phase Flip U_f on the current subset S

        # 2. APPROXIMATE REFLECTION U_ref(E) - Activates the Szegedy "double reflection" walk
        with prw.compute():  # Enter a compute block to allow clean uncomputation of the walk later
            # Repeat the walk application depending on the QPE steps
            for qw_iter in range(len_s):  # Iterate O(1/\sqrt{\delta}) times for Quantum Phase Estimation
            
                # Reflection A (Ref A, on the current node)
                prw.apply(qrw_update.dag(), node_s_ones, node_s_zeros,  # Apply U_U^\dagger (adjoint Update) to un-superimpose edges
                          node_t_ones, node_t_zeros, alpha_ones, alpha_zeros,  # Passing all required registers
                          wstate_ones, wstate_zeros)  # Including the W-states
                for j in range(k):  # Loop over the k elements of the subset S
                    prw.apply(X, wstate_ones[j])  # Apply X gates to effectively flip the control condition for the 0 state
                prw.apply(Z.ctrl(k), qpe_s[qw_iter], wstate_ones)  # Apply a controlled-Z (Phase inversion) conditioned on the QPE current bit
                for j in range(k):  # Loop over the k elements again
                    prw.apply(X, wstate_ones[j])  # Remove the X gates to restore the original state
                prw.apply(qrw_update, node_s_ones, node_s_zeros, node_t_ones,  # Re-apply U_U to recreate the edge superposition (Ref A complete)
                          node_t_zeros, alpha_ones, alpha_zeros, wstate_ones,  # Passing registers
                          wstate_zeros)  # Passing registers

                # Reflection B (Ref B, on the adjacent node, swapping the coordinates)
                prw.apply(qrw_update.dag(), node_s_zeros, node_s_ones,  # Apply U_U^\dagger but with S and S' SWAPPED to reflect on the adjacent vertex
                          node_t_zeros, node_t_ones, alpha_zeros, alpha_ones,  # Notice how node_s_zeros takes the place of node_s_ones
                          wstate_zeros, wstate_ones)  # Notice how wstate_zeros takes the place of wstate_ones
                for j in range(n - k):  # Loop over the n-k elements of the complement group S'
                    prw.apply(X, wstate_zeros[j])  # Apply X gates to flip the control condition
                prw.apply(Z.ctrl(n - k), qpe_s[qw_iter], wstate_zeros)  # Apply the controlled-Z phase inversion on the adjacent Reflection B
                for j in range(k):  # BUG IN ORIGINAL LOGIC: The paper dictates this should restore the n-k elements, not k elements
                    prw.apply(X, wstate_zeros[j])  # Remove the X gates on the complement wstate
                prw.apply(qrw_update, node_s_zeros, node_s_ones, node_t_zeros,  # Re-apply the SWAPPED U_U operator to bounce back (Ref B complete)
                          node_t_ones, alpha_zeros, alpha_ones, wstate_zeros,  # Passing swapped registers
                          wstate_ones)  # Passing swapped registers

            # Clear memory (Reset) and deallocate the temporary variables \alpha and \omega 
            # in order to clean up the quantum gate "thermodynamic waste"
            for j in range(k):  # Loop over the subset S
                prw.apply(  # Uncompute the C-COPY of elements into \alpha
                    qregs_init.copy_register(m).ctrl(), wstate_ones[j],  # Triggered by W-state
                    node_s_ones[j], alpha_ones)  # Deallocate alpha_ones
            for j in range(n - k):  # Loop over the complement S'
                prw.apply(  # Uncompute the C-COPY of elements into \alpha'
                    qregs_init.copy_register(m).ctrl(), wstate_zeros[j],  # Triggered by W-state
                    node_s_zeros[j], alpha_zeros)  # Deallocate alpha_zeros

            prw.apply(generate(k, 1), wstate_ones)  # Uncompute the generation of W-state |Wk>
            prw.apply(generate(n - k, 1), wstate_zeros)  # Uncompute the generation of W-state |W(n-k)>
            
            # Finally, execute the Quantum Fourier Transform (QFT) to convert phase energy 
            # into measurable probability for the simulator (we estimate the end of the Walk)
            prw.apply(QFT(len_s), qpe_s)  # Apply QFT on the Phase Estimation register
            
        # 3. DIFFUSION: Inversion about the mean (Reflect results on the center, purely like Grover)
        for j in range(len_s):  # Iterate over all QPE qubits
            prw.apply(X, qpe_s[j])  # Apply X gates (NOT) to prepare for zero-state reflection
        if len_s > 1:  # If we have more than 1 QPE qubit
            prw.apply(Z.ctrl(len_s - 1), qpe_s)  # Multi-controlled Z gate hitting the zero state
        else:  # If the QPE is only 1 qubit deep
            prw.apply(Z, qpe_s)  # Standard Z gate
        for j in range(len_s):  # Iterate over all QPE qubits again
            prw.apply(X, qpe_s[j])  # Remove the X gates to restore state with flipped mean
            
        # Uncompute for the logical iteration
        prw.uncompute()  # Collapse the entire compute block, physically reverting the QFT and walks without losing the Grover phase

    print("Program qubits")  # Console printout for debugging
    for k, v in prw._qregnames_to_properties.items():  # Loop through all allocated named quantum registers
        print(k, v.slic)  # Print the register name and its size/slices
        
    # Assemble the sub-routines into a real, compact Logical Circuit (Cuccaro/Classical)
    cr = prw.to_circ(link=[classarith, cuccaro_arith])  # Compile the myQLM Program into an executable Circuit
    print(cr.statistics())  # Print the gate count and depth of the generated topology
    
    # Create a processing package ready for Quantum Hardware (Job) by reading the node_s_ones output
    job = cr.to_job(qubits=[*node_s_ones])  # Prepare the final array S for measurement
    
    # If the command line flag dictates it, physically run the entire Job on the QPU simulator
    if to_simulate:  # Check boolean flag
        res = QPU.submit(job)  # Submit the heavy linear algebra simulation to the local CPU processor
        # Print the simulation measurement data: the probabilities for each collapsed state
        for sample in res:  # Loop over all non-zero measured outcomes
            print(sample.probability, sample.state)  # Print percentage and the literal bitstring


# =====================================================================
# SCRIPT EXECUTION BLOCK (Invoked from CLI)
# =====================================================================
if __name__ == '__main__':
    import sys
    # Checks the flag passed by Pytest/CLI to decide whether to block the simulation
    to_simulate = bool(sys.argv[1])
    print(f"To simulate is {to_simulate}")
    
    # The set X (Subset Sum Problem SSP values) 
    values = [1, 2, 3]
    # n = size of X
    n = len(values)
    # k = requested subset size for the constrained Subset-Sum
    k = 2
    # m = log_2(max(X))
    m = max(values).bit_length()
    
    # Target p = target value to reach
    ts = 3
    print(f"n {n}, k {k}, m {m}, values {values}, target sum = {ts}")
    
    # Boot the global MNRS construction
    main(n, k, values, ts, low_width=True, to_simulate=to_simulate)
