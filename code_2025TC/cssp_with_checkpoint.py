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
from qatext.qroutines.datastructure.sliding_sort_array import (
    insert_ld, insert_lw)
from qatext.qroutines.hamming_weight_generate.bartschiE19 import generate
from qatext.utils.qatmgmt.program import ProgramWrapper
from qatext.utils.qatmgmt.routines import QRoutineWrapper

QPU = PyLinalg()


def simulate_and_print_state(prw, n, k, m, len_s, n_qubits_sum, label="State"):
    """
    Compiles and simulates the program up to the current point.
    Prints probability and amplitude for each non-negligible state,
    parsing the raw bitstring into named registers.
    """
    print(f"\n--- Checkpoint: {label} ---")
    import traceback
    try:
        circ = prw.to_circ(link=[classarith, cuccaro_arith])
        qpu = PyLinalg()
        job = circ.to_job()
        res = qpu.submit(job)

        count = 0
        for sample in res:
            if sample.probability > 0.0005:
                count += 1
                raw_str = str(sample.state)[1:-1]

                idx = 0
                dicke_str = raw_str[idx: idx + n]; idx += n
                s1_str = raw_str[idx: idx + k * m]; idx += k * m
                s0_str = raw_str[idx: idx + (n - k) * m]; idx += (n - k) * m
                # skip t_1, t_0, a_1, a_0, w_1, w_0
                idx += k * m + (n - k) * m + 2 * m + k + (n - k)
                qpe_str = raw_str[idx: idx + len_s]; idx += len_s
                sum_str = raw_str[idx: idx + n_qubits_sum]

                amp_str = (f" | Ampl: {sample.amplitude.real:+.4f}{sample.amplitude.imag:+.4f}j"
                           if hasattr(sample, 'amplitude') else "")
                print(f"  Prob: {sample.probability:.5f}{amp_str} | "
                      f"Dicke:{dicke_str} | S_1:{s1_str} | S_0:{s0_str} | "
                      f"QPE:{qpe_str} | Sum:{sum_str}")
        print(f"  Total states: {count}")
    except Exception as e:
        print(f"  Failed: {e}")
        traceback.print_exc()


def update(n, k, m, insert):
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
    insert = insert_lw if low_width else insert_ld
    m = max(values).bit_length()
    delta = n / (k * (n - k))
    len_s = int(np.ceil(np.log2(np.pi / (2 * np.sqrt(delta)))))
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

    # --- Preparation: Dicke state + VBE ---
    prw.apply(generate(n, k), dicke)
    prw.apply(bix.bix_data_compile_time(n, m, k, sorted_values), dicke,
              node_s_ones, node_s_zeros)
    if to_simulate:
        simulate_and_print_state(prw, n, k, m, len_s, n_qubits_sum,
                                 "[1] Preparation (Dicke + VBE)")

    # --- Edge superposition: first application of U_U ---
    qrw_update = update(n, k, m, insert)
    prw.apply(qrw_update, node_s_ones, node_s_zeros, node_t_ones, node_t_zeros,
              alpha_ones, alpha_zeros, wstate_ones, wstate_zeros)
    if to_simulate:
        simulate_and_print_state(prw, n, k, m, len_s, n_qubits_sum,
                                 "[2] Edge Superposition (after U_U)")

    # --- Outer Grover loop ---
    n_external_iters = int(np.ceil(np.sqrt(comb(n, k))))
    for i in range(n_external_iters):

        # Oracle
        qf_ora = oracle(n, k, m, n_qubits_sum, target_sum)
        prw.apply(qf_ora, node_s_ones, sum_reg)
        if to_simulate:
            simulate_and_print_state(prw, n, k, m, len_s, n_qubits_sum,
                                     f"[iter {i+1}] [3] Post-Oracle")

        # QPE-based approximate reflection
        with prw.compute():
            for qw_iter in range(len_s):

                # --- Reflection A ---
                prw.apply(qrw_update.dag(), node_s_ones, node_s_zeros,
                          node_t_ones, node_t_zeros, alpha_ones, alpha_zeros,
                          wstate_ones, wstate_zeros)
                for j in range(k):
                    prw.apply(X, wstate_ones[j])
                prw.apply(Z.ctrl(k), qpe_s[qw_iter], wstate_ones)
                for j in range(k):
                    prw.apply(X, wstate_ones[j])
                prw.apply(qrw_update, node_s_ones, node_s_zeros, node_t_ones,
                          node_t_zeros, alpha_ones, alpha_zeros, wstate_ones,
                          wstate_zeros)
                if to_simulate:
                    simulate_and_print_state(prw, n, k, m, len_s, n_qubits_sum,
                                             f"[iter {i+1}] [4] Post-Ref A (qpe qubit {qw_iter})")

                # --- Reflection B ---
                prw.apply(qrw_update.dag(), node_s_zeros, node_s_ones,
                          node_t_zeros, node_t_ones, alpha_zeros, alpha_ones,
                          wstate_zeros, wstate_ones)
                for j in range(n - k):
                    prw.apply(X, wstate_zeros[j])
                prw.apply(Z.ctrl(n - k), qpe_s[qw_iter], wstate_zeros)
                for j in range(k):
                    prw.apply(X, wstate_zeros[j])
                prw.apply(qrw_update, node_s_zeros, node_s_ones, node_t_zeros,
                          node_t_ones, alpha_zeros, alpha_ones, wstate_zeros,
                          wstate_ones)
                if to_simulate:
                    simulate_and_print_state(prw, n, k, m, len_s, n_qubits_sum,
                                             f"[iter {i+1}] [5] Post-Ref B (qpe qubit {qw_iter})")

            # --- Reset alpha and wstate ---
            for j in range(k):
                prw.apply(
                    qregs_init.copy_register(m).ctrl(), wstate_ones[j],
                    node_s_ones[j], alpha_ones)
            for j in range(n - k):
                prw.apply(
                    qregs_init.copy_register(m).ctrl(), wstate_zeros[j],
                    node_s_zeros[j], alpha_zeros)
            prw.apply(generate(k, 1), wstate_ones)
            prw.apply(generate(n - k, 1), wstate_zeros)
            if to_simulate:
                simulate_and_print_state(prw, n, k, m, len_s, n_qubits_sum,
                                         f"[iter {i+1}] [6] Post-Reset (alpha/wstate)")

            # --- QFT ---
            prw.apply(QFT(len_s), qpe_s)
            if to_simulate:
                simulate_and_print_state(prw, n, k, m, len_s, n_qubits_sum,
                                         f"[iter {i+1}] [7] Post-QFT")

        # --- Diffusion (inversion around zero on QPE register) ---
        for j in range(len_s):
            prw.apply(X, qpe_s[j])
        if len_s > 1:
            prw.apply(Z.ctrl(len_s - 1), qpe_s)
        else:
            prw.apply(Z, qpe_s)
        for j in range(len_s):
            prw.apply(X, qpe_s[j])
        if to_simulate:
            simulate_and_print_state(prw, n, k, m, len_s, n_qubits_sum,
                                     f"[iter {i+1}] [8] Post-Diffusion")

        # --- Uncompute (reverses the compute block) ---
        prw.uncompute()
        if to_simulate:
            simulate_and_print_state(prw, n, k, m, len_s, n_qubits_sum,
                                     f"[iter {i+1}] [9] Post-Uncompute")

    print("Program qubits")
    for k, v in prw._qregnames_to_properties.items():
        print(k, v.slic)
    cr = prw.to_circ(link=[classarith, cuccaro_arith])
    print(cr.statistics())
    job = cr.to_job(qubits=[*node_s_ones])
    if to_simulate:
        res = QPU.submit(job)
        for sample in res:
            print(sample.probability, sample.state)


if __name__ == '__main__':
    import sys
    to_simulate = bool(sys.argv[1])
    print(f"To simulate is {to_simulate}")
    values = [1, 2, 3]
    n = len(values)
    k = 1
    m = max(values).bit_length()
    ts = 3
    print(f"n {n}, k {k}, m {m}, values {values}, target sum = {ts}")
    main(n, k, values, ts, low_width=True, to_simulate=to_simulate)
