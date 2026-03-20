"""
CSSP Diagnostic Script
======================
Tests each component of the quantum walk algorithm in isolation
to identify the source of the flat probability output and linker crashes.

Run each test independently:  python cssp_diagnostic.py test1
  python cssp_diagnostic.py test2
  ...
"""
import sys, os, numpy as np
from math import comb
from qat.lang.AQASM import Program, classarith
from qat.lang.AQASM.gates import H, X, Z
from qat.lang.AQASM.routines import QRoutine
from qat.lang.AQASM.qftarith import QFT
from qat.qpus import PyLinalg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qatext.qroutines import bix, qregs_init
from qatext.qroutines import qregs_init as qregs
from qatext.qroutines.arith import cuccaro_arith
from qatext.qroutines.datastructure.sliding_sort_array import insert_lw, delete
from qatext.qroutines.hamming_weight_generate.bartschiE19 import generate
from qatext.utils.qatmgmt.program import ProgramWrapper
from qatext.utils.qatmgmt.routines import QRoutineWrapper

QPU = PyLinalg()
N, K, M = 3, 1, 2
VALUES = [1, 2, 3]
TARGET = 3
SORTED_VALUES = sorted(VALUES)
N_QUBITS_SUM = int(np.ceil(np.log2(K))) + M


def simulate_and_print(prw, qubits_to_measure, label):
    """Compile circuit, simulate, print results."""
    cr = prw.to_circ(link=[classarith, cuccaro_arith])
    stats = cr.statistics()
    print(f"\n=== {label} ===")
    print(f"  Qubits: {stats['nbqbits']}, Gates: {stats['gate_size']}")
    job = cr.to_job(qubits=[*qubits_to_measure])
    res = QPU.submit(job)
    for sample in res:
        print(f"  P={sample.probability:.6f}  State={sample.state}")
    return res


def build_preparation(prw):
    """Build registers and apply Dicke + VBE + Update preparation."""
    dicke = prw.qarray_alloc(N, 1, "dicke", str)
    s1 = prw.qarray_alloc(K, M, "s_1", int)
    s0 = prw.qarray_alloc(N - K, M, "s_0", int)
    t1 = prw.qarray_alloc(K, M, "t_1", int)
    t0 = prw.qarray_alloc(N - K, M, "t_0", int)
    a1 = prw.qarray_alloc(1, M, "a_1", int)
    a0 = prw.qarray_alloc(1, M, "a_0", int)
    w1 = prw.qarray_alloc(K, 1, "w_1", str)
    w0 = prw.qarray_alloc(N - K, 1, "w_0", str)
    qpe = prw.qarray_alloc(1, 1, "qpe_s", str)
    sum_reg = prw.qarray_alloc(1, N_QUBITS_SUM, "sum", int)

    for qb in qpe:
        prw.apply(H, qb)

    prw.apply(generate(N, K), dicke)
    prw.apply(bix.bix_matrix_compile_time(N, 1, M, K, SORTED_VALUES),
              dicke, s1, s0)

    from cssp import update
    qrw_update = update(N, K, M, insert_lw, delete)
    prw.apply(qrw_update, s1, s0, t1, t0, a1, a0, w1, w0)

    return dicke, s1, s0, t1, t0, a1, a0, w1, w0, qpe, sum_reg, qrw_update


# ============================================================
# TEST 1: Initial state preparation only (no oracle, no walk)
# Expected: uniform distribution over {|1>, |2>, |3>} = 0.333 each
# ============================================================
def test1():
    print("TEST 1: Initial preparation only")
    prw = ProgramWrapper(Program())
    regs = build_preparation(prw)
    s1 = regs[1]
    simulate_and_print(prw, s1, "After Dicke+VBE+Update (no oracle)")


# ============================================================
# TEST 2: Preparation + Oracle only (no walk, no QPE)
# Expected: same probabilities as test1 (oracle only flips phase)
# This confirms oracle doesn't break the state structure
# ============================================================
def test2():
    print("TEST 2: Preparation + Oracle")
    prw = ProgramWrapper(Program())
    regs = build_preparation(prw)
    s1, sum_reg = regs[1], regs[10]

    from cssp import oracle
    qf_ora = oracle(N, K, M, N_QUBITS_SUM, TARGET)
    prw.apply(qf_ora, s1, sum_reg)

    simulate_and_print(prw, s1, "After Oracle (phases flipped, probabilities unchanged)")


# ============================================================
# TEST 3: Test ONE Ref A only (no QPE control, no Ref B)
# This tests if the reflection around the current vertex works
# Expected: if Ref A=identity on the initial state, probabilities unchanged
# ============================================================
def test3():
    print("TEST 3: Preparation + Oracle + Ref A (no QPE control)")
    prw = ProgramWrapper(Program())
    regs = build_preparation(prw)
    s1, s0, t1, t0, a1, a0, w1, w0, qpe, sum_reg, qrw_update = (
        regs[1], regs[2], regs[3], regs[4], regs[5], regs[6],
        regs[7], regs[8], regs[9], regs[10], regs[11])

    from cssp import oracle
    qf_ora = oracle(N, K, M, N_QUBITS_SUM, TARGET)
    prw.apply(qf_ora, s1, sum_reg)

    # Ref A WITHOUT qpe control (unconditional reflection)
    prw.apply(qrw_update.dag(), s1, s0, t1, t0, a1, a0, w1, w0)
    for j in range(K):
        prw.apply(X, w1[j])
    for j in range(N - K):
        prw.apply(X, w0[j])
    # Z on ALL coin qubits (no qpe control)
    prw.apply(Z.ctrl(N - 1), w1, w0)
    for j in range(K):
        prw.apply(X, w1[j])
    for j in range(N - K):
        prw.apply(X, w0[j])
    prw.apply(qrw_update, s1, s0, t1, t0, a1, a0, w1, w0)

    simulate_and_print(prw, s1, "After unconditional Ref A")


# ============================================================
# TEST 4: Preparation + Oracle + unconditional Ref A + Ref B
# Tests full walk step without QPE
# ============================================================
def test4():
    print("TEST 4: Preparation + Oracle + unconditional walk (Ref A + Ref B)")
    prw = ProgramWrapper(Program())
    regs = build_preparation(prw)
    s1, s0, t1, t0, a1, a0, w1, w0, qpe, sum_reg, qrw_update = (
        regs[1], regs[2], regs[3], regs[4], regs[5], regs[6],
        regs[7], regs[8], regs[9], regs[10], regs[11])

    from cssp import oracle
    qf_ora = oracle(N, K, M, N_QUBITS_SUM, TARGET)
    prw.apply(qf_ora, s1, sum_reg)

    # Ref A (unconditional)
    prw.apply(qrw_update.dag(), s1, s0, t1, t0, a1, a0, w1, w0)
    for j in range(K):
        prw.apply(X, w1[j])
    for j in range(N - K):
        prw.apply(X, w0[j])
    prw.apply(Z.ctrl(N - 1), w1, w0)
    for j in range(K):
        prw.apply(X, w1[j])
    for j in range(N - K):
        prw.apply(X, w0[j])
    prw.apply(qrw_update, s1, s0, t1, t0, a1, a0, w1, w0)

    # Ref B (unconditional, with S<->T swap)
    prw.apply(qrw_update.dag(), t1, t0, s1, s0, a1, a0, w1, w0)
    for j in range(K):
        prw.apply(X, w1[j])
    for j in range(N - K):
        prw.apply(X, w0[j])
    prw.apply(Z.ctrl(N - 1), w1, w0)
    for j in range(K):
        prw.apply(X, w1[j])
    for j in range(N - K):
        prw.apply(X, w0[j])
    prw.apply(qrw_update, t1, t0, s1, s0, a1, a0, w1, w0)

    simulate_and_print(prw, s1, "After unconditional walk step (Ref A + Ref B)")


# ============================================================
# TEST 5: Simple Grover on VBE (no walk, no QPE) using
# compute/uncompute INSIDE a QRoutine (like oracle does)
# This tests if the linker handles compute/uncompute with
# generate+bix at the QRoutine level
# ============================================================
def test5():
    print("TEST 5: Grover with VBE diffuser (QRoutine compute/uncompute)")

    # Build diffuser as a QRoutine (like oracle)
    def grover_iteration():
        qrw = QRoutineWrapper(QRoutine())
        dicke = qrw.qarray_wires(N, 1, "dicke", str)
        s1 = qrw.qarray_wires(K, M, "s_1", int)
        s0 = qrw.qarray_wires(N - K, M, "s_0", int)
        sum_reg = qrw.qarray_wires(1, N_QUBITS_SUM, "sum", int)

        with qrw.compute():
            qrw.apply(generate(N, K), dicke)
            qrw.apply(bix.bix_matrix_compile_time(N, 1, M, K, SORTED_VALUES),
                      dicke, s1, s0)

        from cssp import oracle
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

    grover = grover_iteration()
    n_iters = int(np.ceil(np.sqrt(comb(N, K))))
    for _ in range(n_iters):
        prw.apply(grover, dicke, s1, s0, sum_reg)

    # Final prep: also inside a QRoutine to avoid linker clash
    def final_prep():
        qrw = QRoutineWrapper(QRoutine())
        d = qrw.qarray_wires(N, 1, "dicke", str)
        s = qrw.qarray_wires(K, M, "s_1", int)
        sz = qrw.qarray_wires(N - K, M, "s_0", int)
        qrw.apply(generate(N, K), d)
        qrw.apply(bix.bix_matrix_compile_time(N, 1, M, K, SORTED_VALUES),
                  d, s, sz)
        return qrw

    prw.apply(final_prep(), dicke, s1, s0)

    try:
        simulate_and_print(prw, s1, "Grover VBE (QRoutine compute/uncompute)")
    except Exception as e:
        print(f"  LINKER ERROR: {e}")


# ============================================================
# TEST 6: Full original algorithm with 0 Grover iterations
# This confirms the initial state is correct
# ============================================================
def test6():
    print("TEST 6: Full algorithm with 0 Grover iterations")
    from cssp import main
    # Monkey-patch to use 0 iterations
    import cssp
    original_main = cssp.main
    
    prw = ProgramWrapper(Program())
    regs = build_preparation(prw)
    s1 = regs[1]
    simulate_and_print(prw, s1, "0 iterations (initial state)")


if __name__ == '__main__':
    tests = {
        'test1': test1, 'test2': test2, 'test3': test3,
        'test4': test4, 'test5': test5, 'test6': test6,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in tests:
        print(f"Usage: python cssp_diagnostic.py <{'|'.join(tests.keys())}>")
        print("\nTests:")
        print("  test1: Initial state only (uniform?)")
        print("  test2: + Oracle (phases flipped?)")
        print("  test3: + Unconditional Ref A")
        print("  test4: + Unconditional Ref A + Ref B")
        print("  test5: Grover VBE diffuser (linker test)")
        print("  test6: Full algorithm, 0 iterations")
    else:
        tests[sys.argv[1]]()
