"""
myQLM Linker Patch Script
==========================
This script finds, inspects, and patches the myQLM linker's 
_merge_dictionaries._insert function that crashes with KeyError
when the same @build_gate appears in both forward and reversed forms.

Usage on VM:
  python patch_myqlm_linker.py inspect    # Find and show the buggy code
  python patch_myqlm_linker.py patch      # Apply the monkey-patch
  python patch_myqlm_linker.py test       # Test if the patch works
"""
import sys
import os
import importlib
import inspect


def find_myqlm_linking():
    """Find and inspect the myQLM linking modules."""
    modules = {}
    
    # Try to import the relevant modules
    for mod_name in ['qat.core.linking.util', 'qat.lang.linking.util', 
                     'qat.lang.linking.linker']:
        try:
            mod = importlib.import_module(mod_name)
            modules[mod_name] = mod
            filepath = getattr(mod, '__file__', 'unknown')
            print(f"  Found {mod_name} at: {filepath}")
        except ImportError as e:
            print(f"  Cannot import {mod_name}: {e}")
    
    return modules


def inspect_module(modules):
    """Try to inspect the _merge_dictionaries and related functions."""
    core_util = modules.get('qat.core.linking.util')
    if core_util is None:
        print("Cannot find qat.core.linking.util")
        return
    
    print("\n=== Inspecting qat.core.linking.util ===")
    print(f"File: {core_util.__file__}")
    print(f"Type: {'Cython/compiled' if core_util.__file__.endswith('.so') else 'Pure Python'}")
    
    # List all public functions
    print("\nPublic functions/classes:")
    for name in sorted(dir(core_util)):
        if not name.startswith('_'):
            obj = getattr(core_util, name)
            print(f"  {name}: {type(obj).__name__}")
    
    # Try to get source code
    for func_name in ['update_dictionary', '_merge_dictionaries', 'merge_dictionaries']:
        func = getattr(core_util, func_name, None)
        if func:
            print(f"\n=== {func_name} ===")
            try:
                source = inspect.getsource(func)
                print(source[:2000])
            except (OSError, TypeError):
                print(f"  Cannot get source (compiled Cython)")
                print(f"  Signature: {inspect.signature(func) if hasattr(func, '__code__') else 'N/A'}")
                print(f"  Docstring: {func.__doc__}")
    
    # Try to read the .pyx source file if it exists alongside the .so
    so_path = core_util.__file__
    if so_path.endswith('.so'):
        pyx_path = so_path.replace('.so', '.pyx')
        py_path = so_path.rsplit('.', 1)[0] + '.py'
        for src_path in [pyx_path, py_path]:
            if os.path.exists(src_path):
                print(f"\n=== Source file found: {src_path} ===")
                with open(src_path, 'r') as f:
                    content = f.read()
                # Find the _merge_dictionaries and _insert sections
                for keyword in ['_merge_dictionaries', '_insert', 'update_dictionary']:
                    idx = content.find(keyword)
                    if idx >= 0:
                        start = max(0, idx - 200)
                        end = min(len(content), idx + 800)
                        print(f"\n--- Around '{keyword}' ---")
                        print(content[start:end])
                        print("---")
                break
        else:
            # Try to find .pyx in the package directory
            pkg_dir = os.path.dirname(so_path)
            print(f"\nSearching for source files in {pkg_dir}:")
            for f in os.listdir(pkg_dir):
                print(f"  {f}")


def create_monkey_patch():
    """
    Create a monkey-patch for the myQLM linker.
    
    The root cause: when the linker processes a circuit containing both
    a @build_gate gate and its .dag(), the internal anonymous sub-gates
    (named _1, _2, etc.) collide because the forward and reversed versions
    use the same internal names but with different definitions.
    
    Fix: modify _merge_dictionaries to handle conflicts by renaming
    colliding anonymous gates instead of raising KeyError.
    """
    import qat.core.linking.util as core_util
    
    # Save original function
    original_update_dictionary = core_util.update_dictionary
    
    def patched_update_dictionary(circuit, gate_dic, sub_circuit, prefix=""):
        """
        Patched version that handles anonymous gate name collisions
        by renaming conflicting gates with a unique prefix.
        """
        # Strategy: catch KeyError from _merge_dictionaries and 
        # rename conflicting gates in the sub_circuit before merging.
        try:
            return original_update_dictionary(circuit, gate_dic, sub_circuit, prefix)
        except KeyError as e:
            # The KeyError means an anonymous gate name collision.
            # We need to rename the colliding gates in the sub_circuit.
            conflicting_key = str(e).strip("'\"")
            print(f"[PATCH] Resolving gate name collision for '{conflicting_key}'")
            
            # Rename all anonymous gates in the sub_circuit to avoid collision
            _rename_anonymous_gates(sub_circuit, gate_dic)
            
            # Retry with renamed gates
            return original_update_dictionary(circuit, gate_dic, sub_circuit, prefix)
    
    def _rename_anonymous_gates(sub_circuit, existing_dict):
        """Rename anonymous gates in sub_circuit that conflict with existing_dict."""
        if not hasattr(sub_circuit, 'gate_set') or sub_circuit.gate_set is None:
            return
        
        # Find the maximum anonymous index in existing dict
        max_idx = 0
        all_existing_names = set()
        if existing_dict:
            for name in existing_dict:
                all_existing_names.add(name)
                if name.startswith('_') and name[1:].isdigit():
                    max_idx = max(max_idx, int(name[1:]))
        
        # Build rename map for conflicting anonymous gates
        rename_map = {}
        if hasattr(sub_circuit.gate_set, 'items'):
            for name in list(sub_circuit.gate_set.keys()):
                if name in all_existing_names and name.startswith('_'):
                    max_idx += 1
                    new_name = f"_{max_idx}"
                    while new_name in all_existing_names:
                        max_idx += 1
                        new_name = f"_{max_idx}"
                    rename_map[name] = new_name
        
        if not rename_map:
            return
        
        print(f"[PATCH] Renaming gates: {rename_map}")
        
        # Apply renames to gate_set
        new_gate_set = {}
        for name, gate in sub_circuit.gate_set.items():
            new_name = rename_map.get(name, name)
            new_gate_set[new_name] = gate
        
        # Replace gate_set
        # Try dict-like replacement
        try:
            sub_circuit.gate_set.clear()
            sub_circuit.gate_set.update(new_gate_set)
        except (AttributeError, TypeError):
            # If it's a protobuf map, try different approach
            pass
        
        # Apply renames to gate references in the circuit ops
        if hasattr(sub_circuit, 'ops'):
            for op in sub_circuit.ops:
                if hasattr(op, 'gate') and op.gate in rename_map:
                    op.gate = rename_map[op.gate]
    
    # Apply the monkey-patch
    core_util.update_dictionary = patched_update_dictionary
    print("[PATCH] Successfully patched qat.core.linking.util.update_dictionary")
    return True


def test_patch():
    """Test if the patch resolves the linker crash."""
    import numpy as np
    from math import comb
    from qat.lang.AQASM import Program, classarith
    from qat.lang.AQASM.gates import H, X, Z
    from qat.lang.AQASM.routines import QRoutine
    from qat.qpus import PyLinalg
    
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from qatext.qroutines import bix
    from qatext.qroutines.arith import cuccaro_arith
    from qatext.qroutines.hamming_weight_generate.bartschiE19 import generate
    from qatext.utils.qatmgmt.program import ProgramWrapper
    from qatext.utils.qatmgmt.routines import QRoutineWrapper
    from cssp import oracle
    
    N, K, M = 3, 1, 2
    VALUES = [1, 2, 3]
    TARGET = 3
    SORTED_VALUES = sorted(VALUES)
    N_QUBITS_SUM = int(np.ceil(np.log2(K))) + M
    
    # Build the Grover VBE circuit that previously crashed
    def grover_iter():
        qrw = QRoutineWrapper(QRoutine())
        dicke = qrw.qarray_wires(N, 1, "dicke", str)
        s1 = qrw.qarray_wires(K, M, "s_1", int)
        s0 = qrw.qarray_wires(N - K, M, "s_0", int)
        sum_reg = qrw.qarray_wires(1, N_QUBITS_SUM, "sum", int)
        
        with qrw.compute():
            qrw.apply(generate(N, K), dicke)
            qrw.apply(bix.bix_matrix_compile_time(N, 1, M, K, SORTED_VALUES),
                      dicke, s1, s0)
        
        qf_ora = oracle(N, K, M, N_QUBITS_SUM, TARGET)
        qrw.apply(qf_ora, s1, sum_reg)
        qrw.uncompute()
        
        for j in range(N):
            qrw.apply(X, dicke[j])
        qrw.apply(Z.ctrl(N - 1), dicke)
        for j in range(N):
            qrw.apply(X, dicke[j])
        return qrw
    
    prw = ProgramWrapper(Program())
    dicke = prw.qarray_alloc(N, 1, "dicke", str)
    s1 = prw.qarray_alloc(K, M, "s_1", int)
    s0 = prw.qarray_alloc(N - K, M, "s_0", int)
    sum_reg = prw.qarray_alloc(1, N_QUBITS_SUM, "sum", int)
    
    grover = grover_iter()
    n_iters = int(np.ceil(np.sqrt(comb(N, K))))
    for _ in range(n_iters):
        prw.apply(grover, dicke, s1, s0, sum_reg)
    
    # Final preparation
    def final_prep():
        qrw = QRoutineWrapper(QRoutine())
        d = qrw.qarray_wires(N, 1, "dicke", str)
        s = qrw.qarray_wires(K, M, "s_1", int)
        sz = qrw.qarray_wires(N - K, M, "s_0", int)
        qrw.apply(generate(N, K), d)
        qrw.apply(bix.bix_matrix_compile_time(N, 1, M, K, SORTED_VALUES), d, s, sz)
        return qrw
    prw.apply(final_prep(), dicke, s1, s0)
    
    print("\nCompiling circuit (this is where the crash happened)...")
    try:
        cr = prw.to_circ(link=[classarith, cuccaro_arith])
        print(f"SUCCESS! Circuit compiled: {cr.statistics()}")
        
        # Simulate
        QPU = PyLinalg()
        job = cr.to_job(qubits=[*s1])
        res = QPU.submit(job)
        print("\nSimulation results:")
        for sample in res:
            print(f"  P={sample.probability:.6f}  State={sample.state}")
        
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python patch_myqlm_linker.py <inspect|patch|test>")
        print("  inspect - Find and show the myQLM linker code")
        print("  patch   - Apply monkey-patch and test")
        print("  test    - Apply patch + run full Grover VBE test")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == 'inspect':
        print("=== Locating myQLM linking modules ===")
        modules = find_myqlm_linking()
        inspect_module(modules)
    
    elif cmd == 'patch':
        print("=== Applying monkey-patch ===")
        success = create_monkey_patch()
        if success:
            print("\nPatch applied. Now run 'test' to verify.")
    
    elif cmd == 'test':
        print("=== Applying patch and testing ===")
        create_monkey_patch()
        test_patch()
    
    else:
        print(f"Unknown command: {cmd}")
