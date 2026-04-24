"""
CSSP via MNRS Quantum Walk Search -- patched version.

Starting point: the reference code shipped with Lancellotti et al.,
"Solving the Subset Sum Problem via Quantum Walk Search" (IEEE TC, 2026)
in code_2025TC/cssp.py.

The file below differs from the paper code only by the four changes
documented in documentation/report_modifiche/report_modifiche.tex:

  Mod 1 (17 March 2026) -- symmetric Reflection A/B around the full
                           all-zero coin state;
  Mod 2 (20--23 March)  -- removal of the manual alpha/omega cleanup
                           block inside the QPE compute() block;
  Mod 3 (22 March)      -- insert_lw is the default reversible insert
                           variant; delete() in sliding_sort_array.py
                           now calls insert_lw(...).dag() internally;
  Mod 4 (27 March)      -- delta clamped to (0, 1] and len_s clamped
                           to a minimum of 3 QPE qubits.

The direct QFT of the paper is kept: replacing it with its .dag() was
tried during debugging but could not be validated on J(3,1) (the
distribution is flat either way for lack of spectral gap), and the
paper's convention is internally consistent.

The instrumented copy that prints the state at every checkpoint lives
in code_2025TC/cssp_with_checkpoint.py; that file is intentionally
kept out of this "production" module.
"""

from math import comb

import numpy as np

from qat.lang.AQASM import classarith
from qat.lang.AQASM.gates import H, X, Z
from qat.lang.AQASM.program import Program
from qat.lang.AQASM.qftarith import QFT
from qat.lang.AQASM.routines import QRoutine
from qat.qpus import PyLinalg

from qatext.qroutines import bix
from qatext.qroutines import qregs_init
from qatext.qroutines import qregs_init as qregs
from qatext.qroutines.arith import cuccaro_arith
# Mod 3: default to the low-width (reversible) insert; delete() in
# sliding_sort_array.py has been updated to call insert_lw(...).dag().
from qatext.qroutines.datastructure.sliding_sort_array import (
    insert_ld, insert_lw)
from qatext.qroutines.hamming_weight_generate.bartschiE19 import generate
from qatext.utils.qatmgmt.program import ProgramWrapper
from qatext.utils.qatmgmt.routines import QRoutineWrapper

QPU = PyLinalg()


def update(n, k, m, insert):
    """Walk update operator U_U (unchanged from the paper code)."""
    qrw = QRoutineWrapper(QRoutine())

    node_s_ones = qrw.qarray_wires(k, m, "s_1", int)
    node_s_zeros = qrw.qarray_wires(n - k, m, "s_0", int)
    node_t_ones = qrw.qarray_wires(k, m, "t_1", int)
    node_t_zeros = qrw.qarray_wires(n - k, m, "t_0", int)
    alpha_ones = qrw.qarray_wires(1, m, "a_1", int)
    alpha_zeros = qrw.qarray_wires(1, m, "a_0", int)
    wstate_ones = qrw.qarray_wires(k, 1, "w_1", str)
    wstate_zeros = qrw.qarray_wires(n - k, 1, "w_0", str)

    qrout_insert_ones = insert(k, m)
    qrout_insert_zeros = insert(n - k, m)

    qrw.apply(qregs_init.copy_array_of_registers(k, m), node_s_ones,
              node_t_ones)
    qrw.apply(qregs_init.copy_array_of_registers(n - k, m), node_s_zeros,
              node_t_zeros)

    qrw.apply(generate(k, 1), wstate_ones)
    qrw.apply(generate(n - k, 1), wstate_zeros)
    for j in range(k):
        qrw.apply(
            qregs_init.copy_register(m).ctrl(), wstate_ones[j], node_s_ones[j],
            alpha_ones)
    qrw.apply(qrout_insert_ones.dag(), alpha_ones, node_s_ones)
    for j in range(n - k):
        qrw.apply(
            qregs_init.copy_register(m).ctrl(), wstate_zeros[j],
            node_s_zeros[j], alpha_zeros)
    qrw.apply(qrout_insert_zeros.dag(), alpha_zeros, node_s_zeros)

    qrw.apply(qrout_insert_ones, alpha_zeros, node_s_ones)
    qrw.apply(qrout_insert_zeros, alpha_ones, node_s_zeros)
    return qrw


def oracle(n, k, m, n_qubits_sum, target_value):
    """Oracle U_f that flips the phase of subsets summing to target_value
    (unchanged from the paper code)."""
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


def main(n,
         k,
         values: list[int],
         target_sum: int,
         low_width=True,
         to_simulate=False):
    # Mod 3: prefer the low-width (reversible) insert.
    insert = insert_lw if low_width else insert_ld
    m = max(values).bit_length()

    # Mod 4: the Johnson-graph formula delta = n / (k*(n-k)) is designed
    # for k*(n-k) >= n; on small instances (e.g. J(3,1)) it gives values
    # > 1, which makes len_s <= 0 and the QPE block degenerate. Clamp
    # delta to 1 and force at least 3 QPE qubits, so that the rest of
    # the circuit is well-defined numerically even in the regime where
    # MNRS has no quadratic speed-up to offer.
    delta = min(n / (k * (n - k)), 1.0)
    len_s = max(int(np.ceil(np.log2(np.pi / (2 * np.sqrt(delta))))), 3)

    n_qubits_sum = int(np.ceil(np.log2(k))) + m

    sorted_values = sorted(values)
    prw = ProgramWrapper(Program())
    dicke = prw.qarray_alloc(n, 1, "dicke", str)
    node_s_ones = prw.qarray_alloc(k, m, "s_1", int)
    node_s_zeros = prw.qarray_alloc(n - k, m, "s_0", int)
    node_t_ones = prw.qarray_alloc(k, m, "t_1", int)
    node_t_zeros = prw.qarray_alloc(n - k, m, "t_0", int)
    alpha_ones = prw.qarray_alloc(1, m, "a_1", int)
    alpha_zeros = prw.qarray_alloc(1, m, "a_0", int)
    wstate_ones = prw.qarray_alloc(k, 1, "w_1", str)
    wstate_zeros = prw.qarray_alloc(n - k, 1, "w_0", str)

    qpe_s = prw.qarray_alloc(len_s, 1, "qpe_s", str)
    sum_reg = prw.qarray_alloc(1, n_qubits_sum, "sum", int)

    for qb in qpe_s:
        prw.apply(H, qb)

    # Input preparation: Dicke state + vertex binary encoding (VBE).
    prw.apply(generate(n, k), dicke)
    prw.apply(bix.bix_data_compile_time(n, m, k, sorted_values), dicke,
              node_s_ones, node_s_zeros)

    # First application of U_U: build the edge superposition.
    qrw_update = update(n, k, m, insert)
    prw.apply(qrw_update, node_s_ones, node_s_zeros, node_t_ones, node_t_zeros,
              alpha_ones, alpha_zeros, wstate_ones, wstate_zeros)

    n_external_iters = int(np.ceil(np.sqrt(comb(n, k))))
    for _ in range(n_external_iters):
        # Oracle: phase-flip the marked subsets.
        qf_ora = oracle(n, k, m, n_qubits_sum, target_sum)
        prw.apply(qf_ora, node_s_ones, sum_reg)

        # QPE-based approximate reflection U_ref(E).
        with prw.compute():
            for qw_iter in range(len_s):

                # --- Reflection A: reflect around |0...0> on the full coin ---
                # Mod 1: the Z must fire when the WHOLE coin register is in
                # |0>, not only half of it; X gates are applied to both halves
                # symmetrically.
                prw.apply(qrw_update.dag(), node_s_ones, node_s_zeros,
                          node_t_ones, node_t_zeros, alpha_ones, alpha_zeros,
                          wstate_ones, wstate_zeros)
                for j in range(k):
                    prw.apply(X, wstate_ones[j])
                for j in range(n - k):
                    prw.apply(X, wstate_zeros[j])
                prw.apply(Z.ctrl(n), qpe_s[qw_iter],
                          wstate_ones, wstate_zeros)
                for j in range(k):
                    prw.apply(X, wstate_ones[j])
                for j in range(n - k):
                    prw.apply(X, wstate_zeros[j])
                prw.apply(qrw_update, node_s_ones, node_s_zeros, node_t_ones,
                          node_t_zeros, alpha_ones, alpha_zeros, wstate_ones,
                          wstate_zeros)

                # --- Reflection B: reflect around |0...0> on the OTHER edge ---
                # Mod 1: swap S <-> T (v_i <-> v_j) and control Z on the full
                # coin, symmetrically on both halves.
                prw.apply(qrw_update.dag(),
                          node_t_ones, node_t_zeros,
                          node_s_ones, node_s_zeros,
                          alpha_ones, alpha_zeros,
                          wstate_ones, wstate_zeros)
                for j in range(k):
                    prw.apply(X, wstate_ones[j])
                for j in range(n - k):
                    prw.apply(X, wstate_zeros[j])
                prw.apply(Z.ctrl(n), qpe_s[qw_iter],
                          wstate_ones, wstate_zeros)
                for j in range(k):
                    prw.apply(X, wstate_ones[j])
                for j in range(n - k):
                    prw.apply(X, wstate_zeros[j])
                prw.apply(qrw_update,
                          node_t_ones, node_t_zeros,
                          node_s_ones, node_s_zeros,
                          alpha_ones, alpha_zeros,
                          wstate_ones, wstate_zeros)

            # Mod 2: the original cleanup block
            #   for j in range(k):
            #       prw.apply(copy_register(m).ctrl(), w_ones[j], s_ones[j], a_ones)
            #   ...
            #   prw.apply(generate(k, 1),     wstate_ones)
            #   prw.apply(generate(n - k, 1), wstate_zeros)
            # was removed: at this point alpha/omega are in superposition
            # with the visited vertices, so calling generate() as if they
            # were in |0> destroys the walk coherence. AQASM's
            # compute/uncompute already takes care of the correct
            # uncomputation at the end of this block.

            prw.apply(QFT(len_s), qpe_s)

        # Diffusion: inversion around |0...0> on the QPE register.
        for j in range(len_s):
            prw.apply(X, qpe_s[j])
        if len_s > 1:
            prw.apply(Z.ctrl(len_s - 1), qpe_s)
        else:
            prw.apply(Z, qpe_s)
        for j in range(len_s):
            prw.apply(X, qpe_s[j])
        prw.uncompute()

    print("Program qubits")
    for reg_name, reg_props in prw._qregnames_to_properties.items():
        print(reg_name, reg_props.slic)
    cr = prw.to_circ(link=[classarith, cuccaro_arith])
    print(cr.statistics())
    job = cr.to_job(qubits=[*node_s_ones])
    if to_simulate:
        res = QPU.submit(job)
        for sample in res:
            print(sample.probability, sample.state)


if __name__ == '__main__':
    import sys
    to_simulate = (sys.argv[1].lower() == 'true'
                   if len(sys.argv) > 1 else False)
    print(f"To simulate is {to_simulate}")

    values = [1, 2, 3]
    n = len(values)
    k = 1
    m = max(values).bit_length()
    ts = 3
    print(f"n {n}, k {k}, m {m}, values {values}, target sum = {ts}")
    main(n, k, values, ts, low_width=True, to_simulate=to_simulate)
