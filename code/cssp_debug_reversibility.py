
import numpy as np
import sys
from math import comb
from qat.lang.AQASM import classarith
from qatext.qroutines.arith import cuccaro_arith

# QAT imports
from qat.lang.AQASM import Program
from qat.qpus import PyLinalg
from qat.lang.AQASM.gates import H, X
from qat.lang.AQASM.routines import QRoutine
from qatext.utils.qatmgmt.program import ProgramWrapper
from qatext.qroutines import qregs_init
from qatext.qroutines import bix
from qatext.qroutines.datastructure.sliding_sort_array import insert_lw, insert_ld, delete
from qatext.qroutines.hamming_weight_generate.bartschiE19 import generate
import cssp

def debug_reversibility(n=3, k=1, values=[1, 2, 3], m=2):
    print(f"\n=== DEBUG REVERSIBILITY: n={n}, k={k}, values={values} ===")
    
    # Setup
    prog = Program()
    prw = ProgramWrapper(prog)
    
    # Allocate registers
    # Note: Using the exact same allocation order as cssp.py main() to be safe, 
    # though names handle mapping.
    
    dicke = prw.qarray_alloc(n, 1, "dicke", str)
    node_s_ones = prw.qarray_alloc(k, m, "s_1", int)
    node_s_zeros = prw.qarray_alloc(n - k, m, "s_0", int)
    node_t_ones = prw.qarray_alloc(k, m, "t_1", int)
    node_t_zeros = prw.qarray_alloc(n - k, m, "t_0", int)
    alpha_ones = prw.qarray_alloc(1, m, "a_1", int)
    alpha_zeros = prw.qarray_alloc(1, m, "a_0", int)
    wstate_ones = prw.qarray_alloc(k, 1, "w_1", str)
    wstate_zeros = prw.qarray_alloc(n - k, 1, "w_0", str)

    # 1. Initialize System (Uniform Superposition)
    print("STEP 1: Initialization (Dicke + BIX)...")
    prw.apply(generate(n, k), dicke)
    prw.apply(bix.bix_matrix_compile_time(n, 1, m, k, sorted(values)), dicke,
              node_s_ones, node_s_zeros)
    
    # 2. Apply Update
    print("STEP 2: Apply Update Operator...")
    # Using low_width=True for insert as commonly used
    qrw_update = cssp.update(n, k, m, insert_lw, delete)
    prw.apply(qrw_update, node_s_ones, node_s_zeros, node_t_ones, node_t_zeros,
              alpha_ones, alpha_zeros, wstate_ones, wstate_zeros)
              
    # 3. Apply Update DAG (Immediate Uncompute)
    print("STEP 3: Apply Update DAG (Reverse)...")
    prw.apply(qrw_update.dag(), node_s_ones, node_s_zeros, node_t_ones, node_t_zeros,
              alpha_ones, alpha_zeros, wstate_ones, wstate_zeros)

    # 4. Simulation
    print("STEP 4: Running Simulation (PyLinalg)...")
    sys.stdout.flush()
    qpu = PyLinalg()
    # CRITICAL: Link arithmetic libraries
    circuit = prog.to_circ(link=[classarith, cuccaro_arith])
    print(f"Circuit Size: {circuit.nbqbits} qubits, {len(circuit.ops)} gates")
    sys.stdout.flush()
    job = circuit.to_job()
    result = qpu.submit(job)
    print("STEP 4.5: Simulation Completed.")
    sys.stdout.flush()
    
    # 5. Check Results
    print("STEP 5: Analyzing Registers...")
    
    N_search_space = comb(n, k)
    print(f"Expected Search Space Size: {N_search_space}")
    
    garbage_ancillas = False
    dist_s = {}
    
    total_prob = 0.0
    
    for sample in result:
        prob = sample.probability
        total_prob += prob
        if prob < 1e-10: continue

        # Extract register values
        # We check if ancillas are non-zero
        # Ancillas: t_1, t_0, a_1, a_0, w_1, w_0
        
        # Note: result[register_name] might return integer value
        # But iterate loop gives ResultEntry which usually allows access by name if mapped
        
        # Let's try to read bits directly.
        # We assume the user wants to see if ANY garbage exists.
        
        # In PyLinalg: sample.state is an intermediate object. 
        # But we can look at the parsed values if we used a simulator that supports it.
        # PyLinalg usually returns outcomes where we have to parse bits ourself or rely on QLM to do it.
        # But `sample` object usually has a `value` field if measured?
        # No, we simulated the full state vector (implied by default for small circuits).
        # We need to extract values for each register.
        
        # Helper to get value of a register from the state int
        def get_reg_val(reg_obj, state_int):
            # reg_obj is a QRegister, has .start and .length (or .size)
            # wait, 'prw.qarray_alloc' returns a valid object that might not have direct start index in final circuit
            # because 'prw' wraps it.
            # But the 'circuit' object has a valid .qregs list.
            pass
            
        # Let's just assume we can deduce it from the variable names.
        # Or easier: print non-zero ancillas using the built-in state dump if available.
        # But we want automated checking.
        
        # Let's use the `qv` (QRegister) objects from allocation.
        # prw.qarray_alloc returns list of Qbits.
        # Effectively we need their indices.
        
        pass

    # Alternative: Use sample.unwrap() if available?
    # Let's use the provided `cssp.py` style or just use qat-core features.
    # We will use result to loop and check qubit indices.
    
    # Map qbits to registers
    # dicke: 0..n-1
    # etc...
    # Since we allocated sequentially using prw, the qubit indices should be sequential.
    
    # Let's rebuild the offsets.
    current_idx = 0
    regs = {}
    
    def alloc_track(name, size):
        nonlocal current_idx
        # size in qubits
        indices = list(range(current_idx, current_idx + size))
        regs[name] = indices
        current_idx += size
        
    # Re-calculate sizes
    alloc_track("dicke", n * 1)
    alloc_track("s_1", k * m)
    alloc_track("s_0", (n-k) * m)
    alloc_track("t_1", k * m)
    alloc_track("t_0", (n-k) * m)
    alloc_track("a_1", 1 * m)
    alloc_track("a_0", 1 * m)
    alloc_track("w_1", k * 1)
    alloc_track("w_0", (n-k) * 1)
    
    bad_states_count = 0
    
    print(f"\nScanning {len(result)} states...")
    
    for sample in result:
        if sample.probability < 1e-6: continue
        
        state_int = sample.state.lsb_int
        
        def extract_val(name):
            indices = regs[name]
            val = 0
            for i, bit_idx in enumerate(indices):
                if (state_int >> bit_idx) & 1:
                    val |= (1 << i)
            return val
            
        # Check Ancillas
        t1 = extract_val("t_1")
        t0 = extract_val("t_0")
        a1 = extract_val("a_1")
        a0 = extract_val("a_0")
        w1 = extract_val("w_1")
        w0 = extract_val("w_0")
        
        is_garbage = (t1 != 0) or (t0 != 0) or (a1 != 0) or (a0 != 0) or (w1 != 0) or (w0 != 0)
        
        if is_garbage:
            garbage_ancillas = True
            bad_states_count += 1
            print(f"[GARBAGE] State: {state_int} Prob: {sample.probability:.4f}")
            print(f"  Expected 0 for: t1={t1}, t0={t0}, a1={a1}, a0={a0}, w1={w1}, w0={w0}")
            
        else:
            # Check Node S distribution
            # We treat 's_1' as the subset.
            # Ideally we check what elements are in it.
            # But just storing the integer value is enough to check uniformity.
            s1_val = extract_val("s_1")
            dist_s[s1_val] = dist_s.get(s1_val, 0) + sample.probability

    print("\n=== SUMMARY ===")
    if garbage_ancillas:
        print(f"FAIL: Found {bad_states_count} states with garbage in ancillas!")
    else:
        print("PASS: No garbage in ancillary registers (t, a, w).")
        
    print("\nDistribution of S (should be uniform over valid subsets):")
    for s_val, prob in dist_s.items():
        # Ideally decode s_val to elements
        print(f"  S_val={s_val}: {prob:.4f}")
        
    # Check uniformity
    probs = np.array(list(dist_s.values()))
    mean_prob = np.mean(probs)
    std_dev = np.std(probs)
    print(f"  Mean Prob: {mean_prob:.4f}, Std Dev: {std_dev:.4f}")
    
    if std_dev > 1e-5:
        print("WARNING: Distribution is NOT uniform!")
    else:
        print("SUCCESS: Distribution is uniform.")

    if garbage_ancillas:
        sys.exit(1)

if __name__ == "__main__":
    debug_reversibility()
