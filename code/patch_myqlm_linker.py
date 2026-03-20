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
    Patch update_dictionary to PREVENT gate name collisions by pre-renaming
    all anonymous gates in each subcircuit with unique prefixes BEFORE merging.
    
    The Cython _merge_dictionaries._insert crashes when the same anonymous
    gate name (_1, _2, etc.) exists in both dictionaries with different
    definitions. By giving each subcircuit's anonymous gates globally unique
    names, collisions become impossible.
    """
    import qat.core.linking.util as core_util
    
    original_update_dictionary = core_util.update_dictionary
    _call_counter = [0]
    
    def _rename_anonymous_recursive(gate_dic, prefix):
        """
        Recursively rename all anonymous gates (_N) in a gate dictionary
        to use a unique prefix (e.g., _N -> _p3_N).
        Returns the rename map applied at this level.
        """
        if not gate_dic:
            return {}
        
        # Build rename map for this level
        rename_map = {}
        for name in list(gate_dic.keys()):
            if isinstance(name, str) and name.startswith('_') and not name.startswith('_p'):
                new_name = f"{prefix}{name}"
                rename_map[name] = new_name
        
        if not rename_map:
            return {}
        
        # Apply renames: rebuild the dictionary with new keys
        items = []
        for name in list(gate_dic.keys()):
            new_name = rename_map.get(name, name)
            items.append((new_name, gate_dic[name]))
        
        # Clear and repopulate
        for k in list(gate_dic.keys()):
            del gate_dic[k]
        for k, v in items:
            gate_dic[k] = v
        
        # Update internal references in each gate definition
        for name in list(gate_dic.keys()):
            gate_def = gate_dic[name]
            if hasattr(gate_def, 'circuit') and gate_def.circuit is not None:
                circ = gate_def.circuit
                # Rename op references
                if hasattr(circ, 'ops'):
                    for op in circ.ops:
                        if hasattr(op, 'gate') and op.gate in rename_map:
                            op.gate = rename_map[op.gate]
                # Recursively rename nested gate_set
                if hasattr(circ, 'gate_set') and circ.gate_set:
                    nested_map = _rename_anonymous_recursive(circ.gate_set, prefix)
                    # Update ops with nested renames too
                    if nested_map and hasattr(circ, 'ops'):
                        for op in circ.ops:
                            if hasattr(op, 'gate') and op.gate in nested_map:
                                op.gate = nested_map[op.gate]
            
            # Handle syntax references if present
            if hasattr(gate_def, 'syntax') and gate_def.syntax is not None:
                if hasattr(gate_def.syntax, 'name') and gate_def.syntax.name in rename_map:
                    gate_def.syntax.name = rename_map[gate_def.syntax.name]
        
        return rename_map
    
    def patched_update_dictionary(circuit, subcircuit_gate_dic):
        """
        Pre-rename all anonymous gates in the subcircuit to prevent
        collisions with the circuit's gate dictionary.
        """
        _call_counter[0] += 1
        prefix = f"_p{_call_counter[0]}"
        
        # Only rename if there could be conflicts
        if circuit.gate_set and subcircuit_gate_dic:
            _rename_anonymous_recursive(subcircuit_gate_dic, prefix)
        
        return original_update_dictionary(circuit, subcircuit_gate_dic)
    
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
