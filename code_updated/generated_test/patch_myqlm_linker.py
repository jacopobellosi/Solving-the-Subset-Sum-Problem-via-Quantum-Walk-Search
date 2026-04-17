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
    
    # Inspect GateSet type
    print("\n=== Inspecting GateSet type ===")
    try:
        from qat.comm.datamodel.ttypes import GateDictionary
        gs = GateDictionary()
        print(f"GateDictionary type: {type(gs)}")
        print(f"GateDictionary dir: {[x for x in dir(gs) if not x.startswith('__')]}")
    except ImportError:
        pass
    
    try:
        # Build a tiny circuit to inspect its gate_set type
        from qat.lang.AQASM import Program
        from qat.lang.AQASM.gates import H
        p = Program()
        q = p.qalloc(1)
        p.apply(H, q)
        c = p.to_circ()
        print(f"\nCircuit.gate_set type: {type(c.gate_set)}")
        print(f"gate_set dir: {[x for x in dir(c.gate_set) if not x.startswith('__')]}")
        print(f"gate_set len: {len(c.gate_set) if hasattr(c.gate_set, '__len__') else 'N/A'}")
        if hasattr(c.gate_set, '__iter__'):
            print(f"gate_set iterable: True")
            for name in c.gate_set:
                print(f"  gate name: {name}, type: {type(name)}")
        if hasattr(c.gate_set, '__getitem__'):
            print(f"gate_set subscriptable: True")
    except Exception as e:
        print(f"Error inspecting: {e}")


def create_monkey_patch():
    """
    Patch at the highest possible level to prevent gate name collisions.
    
    Strategy: Patch `link_gates` in `qat.lang.linking.util` (the entry point
    called by `Linker.link`). Before calling the original, we pre-process
    the circuit's entire gate dictionary to make all anonymous gate names
    globally unique, preventing any collision in the Cython merge code.
    """
    import qat.lang.linking.util as lang_util
    import qat.core.linking.util as core_util
    
    original_link_gates = lang_util.link_gates
    _counter = [0]
    
    def _make_all_anonymous_unique(gate_dic, visited=None):
        """
        Walk the ENTIRE gate dictionary tree and rename all anonymous
        gates to globally unique names. This must happen BEFORE any
        Cython merge code runs.
        """
        if not gate_dic:
            return
        if visited is None:
            visited = set()
        
        for name in list(gate_dic):
            gate_def = gate_dic[name]
            gd_id = id(gate_def)
            if gd_id in visited:
                continue
            visited.add(gd_id)
            
            if hasattr(gate_def, 'circuit') and gate_def.circuit is not None:
                circ = gate_def.circuit
                if hasattr(circ, 'gate_set') and circ.gate_set:
                    # Rename anonymous gates in this sub-circuit
                    _counter[0] += 1
                    prefix = f"_u{_counter[0]}"
                    
                    rename_map = {}
                    for sub_name in list(circ.gate_set):
                        if (isinstance(sub_name, str) and sub_name.startswith('_') 
                            and not sub_name.startswith('_u')):
                            new_name = f"{prefix}{sub_name}"
                            rename_map[sub_name] = new_name
                    
                    if rename_map:
                        # Rebuild gate_set with new names
                        items = [(rename_map.get(k, k), circ.gate_set[k]) 
                                 for k in list(circ.gate_set)]
                        for k in list(circ.gate_set):
                            del circ.gate_set[k]
                        for k, v in items:
                            circ.gate_set[k] = v
                        
                        # Update op references
                        if hasattr(circ, 'ops'):
                            for op in circ.ops:
                                if hasattr(op, 'gate') and op.gate in rename_map:
                                    op.gate = rename_map[op.gate]
                    
                    # Recurse into sub-gate definitions
                    _make_all_anonymous_unique(circ.gate_set, visited)
    
    def patched_link_gates(circuit, *args, **kwargs):
        """
        Pre-process the circuit to make all anonymous gate names unique,
        then call the original link_gates.
        """
        if hasattr(circuit, 'gate_set') and circuit.gate_set:
            _make_all_anonymous_unique(circuit.gate_set)
        return original_link_gates(circuit, *args, **kwargs)
    
    # Patch link_gates at the module level
    lang_util.link_gates = patched_link_gates
    
    # Also try to patch the reference used by Linker.link_gates (the class method)
    try:
        import qat.lang.linking.linker as linker_mod
        original_linker_link_gates = linker_mod.Linker.link_gates
        
        def patched_linker_link_gates(self, circuit, *args, **kwargs):
            if hasattr(circuit, 'gate_set') and circuit.gate_set:
                _make_all_anonymous_unique(circuit.gate_set)
            return original_linker_link_gates(self, circuit, *args, **kwargs)
        
        linker_mod.Linker.link_gates = patched_linker_link_gates
        print("[PATCH] Patched Linker.link_gates")
    except Exception as e:
        print(f"[PATCH] Could not patch Linker.link_gates: {e}")
    
    # Also patch update_dictionary in BOTH modules
    original_update_dict = core_util.update_dictionary
    def patched_update_dict(circuit, subcircuit_gate_dic):
        _counter[0] += 1
        prefix = f"_u{_counter[0]}"
        if subcircuit_gate_dic:
            rename_map = {}
            for name in list(subcircuit_gate_dic):
                if (isinstance(name, str) and name.startswith('_')
                    and not name.startswith('_u')):
                    new_name = f"{prefix}{name}"
                    rename_map[name] = new_name
            if rename_map:
                items = [(rename_map.get(k, k), subcircuit_gate_dic[k]) 
                         for k in list(subcircuit_gate_dic)]
                for k in list(subcircuit_gate_dic):
                    del subcircuit_gate_dic[k]
                for k, v in items:
                    subcircuit_gate_dic[k] = v
                    if hasattr(v, 'circuit') and v.circuit is not None:
                        if hasattr(v.circuit, 'ops'):
                            for op in v.circuit.ops:
                                if hasattr(op, 'gate') and op.gate in rename_map:
                                    op.gate = rename_map[op.gate]
                        if hasattr(v.circuit, 'gate_set') and v.circuit.gate_set:
                            nested_items = [(rename_map.get(g, g), v.circuit.gate_set[g])
                                           for g in list(v.circuit.gate_set)]
                            for g in list(v.circuit.gate_set):
                                del v.circuit.gate_set[g]
                            for g, gv in nested_items:
                                v.circuit.gate_set[g] = gv
        return original_update_dict(circuit, subcircuit_gate_dic)
    
    core_util.update_dictionary = patched_update_dict
    # Try to also patch the lang module's reference
    if hasattr(lang_util, 'update_dictionary'):
        lang_util.update_dictionary = patched_update_dict
    
    print("[PATCH] Patched update_dictionary in core and lang modules")
    print("[PATCH] All patches applied successfully")
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
