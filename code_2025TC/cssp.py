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
from qatext.qroutines.datastructure.sliding_sort_array import (  # ld stands for low-depth
    insert_ld, insert_lw)
from qatext.qroutines.hamming_weight_generate.bartschiE19 import generate
from qatext.utils.qatmgmt.program import ProgramWrapper
from qatext.utils.qatmgmt.routines import QRoutineWrapper

QPU = PyLinalg()


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
    # Assuming no duplicates
    m = max(values).bit_length()
    # the spectral gap of the johnson graph (n, k)
    delta = n / (k * (n - k))
    # 2^s >  \pi/(2 \sqrt(delta)) -> s > log_2(\pi/(2\sqrt(\delta)))
    len_s = int(np.ceil(np.log2(np.pi / (2 * np.sqrt(delta)))))
    # I need to store the sum of k elements, each one having size m qubits
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
    # a, b -> a+b, b
    for qb in qpe_s:
        prw.apply(H, qb)

    # dicke + bix
    prw.apply(generate(n, k), dicke)
    prw.apply(bix.bix_data_compile_time(n, m, k, sorted_values), dicke,
              node_s_ones, node_s_zeros)
    qrw_update = update(n, k, m, insert)
    prw.apply(qrw_update, node_s_ones, node_s_zeros, node_t_ones, node_t_zeros,
              alpha_ones, alpha_zeros, wstate_ones, wstate_zeros)

    # n iterations external
    n_external_iters = int(np.ceil(np.sqrt(comb(n, k))))
    for _ in range(n_external_iters):
        # oracle
        qf_ora = oracle(n, k, m, n_qubits_sum, target_sum)
        prw.apply(qf_ora, node_s_ones, sum_reg)

        # walk
        with prw.compute():
            for qw_iter in range(len_s):
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

                # ref b
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

            # reset alpha_0/1
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
            prw.apply(QFT(len_s), qpe_s)
        # inversion around zero
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
