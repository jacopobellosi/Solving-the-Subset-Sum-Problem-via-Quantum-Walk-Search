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
    Monkey-patch _merge_dictionaries to handle anonymous gate name collisions.
    
    Root cause: when the linker processes both forward and .dag() versions
    of a @build_gate gate, internal anonymous sub-gates (_1, _2, etc.) 
    collide because both versions use the same names with different definitions.
    
    Fix: patch _merge_dictionaries to rename colliding keys instead of crashing.
    """
    import qat.core.linking.util as core_util
    
    original_merge = core_util._merge_dictionaries
    
    def patched_merge_dictionaries(gate_dic_1, gate_dic_2, full_var_dic=None):
        """
        Patched version: if key collisions occur, rename ALL conflicting
        gates in gate_dic_2 before merging. Retries in a loop.
        """
        max_retries = 50
        for attempt in range(max_retries):
            try:
                return original_merge(gate_dic_1, gate_dic_2, full_var_dic)
            except KeyError as e:
                conflicting_key = str(e).strip("'\"")
                
                # Find max anonymous index across both dicts
                max_idx = 0
                existing_names = set()
                for dic in [gate_dic_1, gate_dic_2]:
                    if dic:
                        for name in dic:
                            existing_names.add(name)
                            if isinstance(name, str) and name.startswith('_') and name[1:].isdigit():
                                max_idx = max(max_idx, int(name[1:]))
                
                # Rename ALL conflicting gates in dic_2 (not just the one that crashed)
                rename_map = {}
                if gate_dic_2:
                    for name in list(gate_dic_2.keys()):
                        if name in gate_dic_1:  # Conflict with dic_1
                            max_idx += 1
                            new_name = f"_{max_idx}"
                            while new_name in existing_names or new_name in rename_map.values():
                                max_idx += 1
                                new_name = f"_{max_idx}"
                            rename_map[name] = new_name
                            existing_names.add(new_name)
                
                if not rename_map:
                    raise  # Can't fix
                
                # Apply renames to gate_dic_2
                items = [(rename_map.get(k, k), gate_dic_2[k]) for k in list(gate_dic_2.keys())]
                for k in list(gate_dic_2.keys()):
                    del gate_dic_2[k]
                for k, v in items:
                    gate_dic_2[k] = v
                    # Rename internal op references
                    if hasattr(v, 'circuit') and v.circuit is not None:
                        for op in v.circuit.ops:
                            if hasattr(op, 'gate') and op.gate in rename_map:
                                op.gate = rename_map[op.gate]
                        if hasattr(v.circuit, 'gate_set') and v.circuit.gate_set:
                            nested = [(rename_map.get(g, g), v.circuit.gate_set[g]) 
                                     for g in list(v.circuit.gate_set.keys())]
                            for g in list(v.circuit.gate_set.keys()):
                                del v.circuit.gate_set[g]
                            for g, gv in nested:
                                v.circuit.gate_set[g] = gv
                
                # Also update any NEW items that reference renamed gates
                for k in list(gate_dic_2.keys()):
                    v = gate_dic_2[k]
                    if hasattr(v, 'circuit') and v.circuit is not None:
                        for op in v.circuit.ops:
                            if hasattr(op, 'gate') and op.gate in rename_map:
                                op.gate = rename_map[op.gate]
        
        raise RuntimeError(f"Could not resolve gate collisions after {max_retries} attempts")
    
    core_util._merge_dictionaries = patched_merge_dictionaries
    print("[PATCH] Successfully patched qat.core.linking.util._merge_dictionaries")
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
