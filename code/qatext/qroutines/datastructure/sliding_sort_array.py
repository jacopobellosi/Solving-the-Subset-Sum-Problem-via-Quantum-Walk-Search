from qat.lang.AQASM.gates import CNOT, SWAP, X
from qat.lang.AQASM.misc import build_gate
from qat.lang.AQASM.qint import QInt
from qat.lang.AQASM.routines import QRoutine
from qatext.qroutines.qregs_init import copy_register
from qatext.qroutines.qubitshuffle.rotate import swap_qreg_cells
from qatext.utils.qatmgmt.routines import QRoutineWrapper


@build_gate('SLIDING_SORT_INSERT', [int, int], lambda n, m: n * m + m)
def insert_ld(n, m):
    """n cells, each one of size m.
    Expect qregs in this order: X (element to insert), A (the register T)
    """
    qf = QRoutine()
    qr_val = qf.new_wires(m, QInt)  # The element \alpha to insert
    qrs_a = []
    # Register T (size n)
    for _ in range(n):
        qrs_a.append(qf.new_wires(m, QInt))
    # Register T1 (size n): First ancillary register
    qrs_ai = []
    for _ in range(n):
        _qr = qf.new_wires(m, QInt)
        qrs_ai.append(_qr)
        qf.set_ancillae(_qr)
    # Register T2 (size n): Second ancillary register (holds comparator bool results)
    qr_aii = qf.new_wires(n, QInt)
    qf.set_ancillae(qr_aii)
    # STAGE 1: FAN-OUT
    # "First, the value contained in \alpha is copied into all the cells of T1..."
    for i in range(n):
        for qb1, qb2 in zip(qr_val, qrs_ai[i]):
            qf.apply(CNOT, qb1, qb2)
    # "...and into the last cell of T. (depth of log k using fan-out)"
    for qb1, qb2 in zip(qr_val, qrs_a[n - 1]):
        qf.apply(CNOT, qb1, qb2)
    # STAGE 2: COMPARE
    # "Each cell of T is compared with T1... storing an all-ones string in T2 if T >= T1"
    for qr_a, qr_ai, qb_aii in zip(qrs_a, qrs_ai, qr_aii):
        (qr_ai <= qr_a).evaluate(output=qb_aii)  # Result written to T2 (qb_aii)
    # STAGE 3: CONDITIONAL SWAP 1
    # "apply k conditional swaps between the cell T|i and T|i+1 if T2|i is all-ones"
    for i in range(n - 1):
        qr_a, qr_ai, qb_aii = qrs_a[i], qrs_ai[i + 1], qr_aii[i]
        for qb_a, qb_ai in zip(qr_a, qr_ai):
            qf.apply(SWAP.ctrl(), qb_aii, qb_ai, qb_a)
    # STAGE 4: CONDITIONAL SWAP 2
    # "A second layer of conditional swaps, controlled again by T2|i, is then applied between T|i+1 and T1|i+1"
    for i in range(n - 1):
        qr_a, qr_ai, qb_aii = qrs_a[i + 1], qrs_ai[i + 1], qr_aii[i]
        for qb_a, qb_ai in zip(qr_a, qr_ai):
            qf.apply(SWAP.ctrl(), qb_aii, qb_a, qb_ai)
    # STAGES 5 & 6: UNCOMPUTE T1 AND T2
    # "The last two stages resets the cells of T2 and T1 to zero by performing the second stage and first stage again."
    # Stage 5 (Uncompute Compare / Reset T2)
    for qr_a, qr_ai, qb_aii in zip(qrs_a, qrs_ai, qr_aii):
        (qr_ai <= qr_a).evaluate(output=qb_aii)
    # Stage 6 (Uncompute Fan-Out / Reset T1 using X.ctrl acting as un-CNOT)
    for i in range(n):
        for qb1, qb2 in zip(qr_val, qrs_ai[i]):
            qf.apply(X.ctrl(), qb1, qb2)

    return qf


@build_gate('SLIDING_SORT_DELETE', [int, int], lambda n, m: n * m + m)
def delete(n, m):
    qf = QRoutine()
    qw = qf.new_wires(n * m + m)
    qf1 = insert_ld(n, m).dag()
    qf.apply(qf1, *qw)
    return qf


@build_gate('SLIDING_SORT_INSERT', [int, int], lambda n, m: n * m + m)
def insert_lw(n, m):
    """Low-Width insert.
    N cells, each one of size m.
    Expect qregs in this order: X, A
    """
    qrw = QRoutineWrapper(QRoutine())
    qr_val = qrw.qarray_wires(1, m, "X", int)  # The value \alpha to be inserted
    qarray = qrw.qarray_wires(n, m, "A", int)  # Register T (size n cells)
    # Paper Step: "utilizes only three additional ancillae qubits... single qubit register r"
    # Here we allocate just 1 ancillary wire for the comparison result (acting as 'r').
    # The comparators themselves manage the other 2 internal ancillae implicitly.
    qr_out = qrw.new_wires(1)
    qrw.set_ancillae(qr_out)
    qrw.qarray_wires_noalloc(1, 1, "out", qr_out[0].index, str, False)
    # Paper Step: "first copies the value stored in \alpha into the last cell of T"
    # This prepares the array by placing the element in the empty slot at the end.
    qrw.apply(copy_register(m), qr_val, qarray[-1])
    # Paper Step: "then performs k-1 identical stages. Each stage i, with 1 <= i < k"
    # We iterate sequentially backwards through the array from the second-to-last element to index 0.
    for j in range(n - 2, -1, -1):
        # SUB-STEP 1: COMPARE
        # "compare the value contained in T|k-i-1 with the value contained in \alpha... XOR the state |1> in register r"
        (qarray[j] >= qr_val[0]).evaluate(output=qr_out)
        # SUB-STEP 2: CONTROLLED SWAP
        # "using the value in r as control, we perform a controlled swap operation between T|k-i and T|k-i-1"
        # Since r (qr_out) is True only if T[current] >= \alpha, this swaps \alpha into the lower index cell dynamically.
        qrw.apply(swap_qreg_cells(m).ctrl(), qr_out, qarray[j], qarray[j + 1])
        # SUB-STEP 3: UNCOMPUTE COMPARATOR
        # "To reuse the same ancillary qubit r in the next stage, we perform the comparison 
        # T|... = \alpha... bringing back its state to |0>"
        # By re-evaluating the same condition, we uncompute 'qr_out', freeing it cleanly for the next loop iteration.
        (qarray[j] >= qr_val[0]).evaluate(output=qr_out)
    return qrw._qroutine
