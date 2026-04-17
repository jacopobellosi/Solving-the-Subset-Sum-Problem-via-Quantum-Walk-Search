from test.common_pytest import (REVERSIBLE_ON, REVERSIBLE_ON_REASON,
                                CircuitTestHelpers)

import numpy as np
import pytest
import qat.lang.AQASM.classarith
from qat.lang.AQASM.program import Program
from qatext.qpus.reversible import get_states_from_program_wrapper
from qatext.qroutines import qregs_init as qregs
from qatext.qroutines.datastructure.sliding_sort_array import insert_lw as insert
from qatext.utils.bits.conversion import (get_int_from_bitarray,
                                          get_ints_from_bitarray)
from qatext.utils.qatmgmt.program import ProgramWrapper

@pytest.mark.usefixtures("setup_simulator", "setup_logger")
class TestLowWidthSlidingSort(CircuitTestHelpers):

    @pytest.mark.parametrize(
        "values, max_bits, value_to_insert",
        [
            # Insert in the middle
            ([1, 2, 4], 4, 3),
            ([2, 4, 6], 7, 3),
            # Insert at the beginning
            ([2, 3, 4], 5, 1),
            # Insert at the end
            ([1, 3, 4], 5, 5),
            # Insert duplicate in the middle
            ([1, 2, 3], 3, 2),
            # Insert duplicate at the end
            ([1, 2, 3], 3, 3),
            # Insert into empty list
            ([], 5, 2),
            # Insert below lower bound
            ([1, 2, 3], 3, 0),
            # Insert above upper bound
            ([1, 2, 3], 3, 4),
            # Single-element list, insert before
            ([3], 3, 2),
            # Single-element list, insert after
            ([2], 4, 3),
            # 0-element list, insert after
            ([], 4, 3),
            # Insertion of existing max value
            ([1, 2], 3, 3),
        ])
    @pytest.mark.skipif(not REVERSIBLE_ON, reason=REVERSIBLE_ON_REASON)
    def test_lw_insertion(self, values, max_bits, value_to_insert):
        m = max_bits
        # last one is the empty cell, used as temporary
        n = len(values) + 1
        prw = ProgramWrapper(Program())
        
        qr_x = prw.qarray_alloc(1, m, "x", int)
        qfun = qregs.initialize_qureg_given_int(value_to_insert, m, False)
        prw.apply(qfun, qr_x)

        qrs_data = prw.qarray_alloc(n, m, "a", int)
        for i, value in enumerate(values):
            qfun = qregs.initialize_qureg_given_int(value, m, False)
            prw.apply(qfun, qrs_data[i])
            
        # insert_lw requires a single ancilla block `qr_out`, not the massive a1 and a2 blocks
        # We calculate the next available wire precisely to avoid IndexErrors on empty arrays
        next_wire = qrs_data[-1].start + qrs_data[-1].length
        prw.qarray_noalloc(1, 1, "qr_out", next_wire, int)
        prw.qarray_noalloc(None, None, "anc", next_wire + 1, str, unknown_size=True)

        qf = insert(n, m)
        prw.apply(qf, qr_x, *qrs_data)

        res = get_states_from_program_wrapper(prw, [qat.lang.AQASM.classarith])

        x_val = get_int_from_bitarray(res['x'], False)
        # For completely empty arrays, we still have 1 cell (`n=1`) dedicated to the new element
        if len(values) == 0:
            a_vals = get_ints_from_bitarray(res['a'], n, m, False)
            qr_out_vals = (0,)
        else:
            a_vals = get_ints_from_bitarray(res['a'], n, m, False)
            qr_out_vals = get_ints_from_bitarray(res['qr_out'], 1, 1, False)
        
        ax_val = res['anc']

        values.append(value_to_insert)
        assert (x_val == value_to_insert)
        assert (tuple(sorted(values)) == a_vals)
        # Check that the low-width ancilla uncomputes correctly to zero
        assert (qr_out_vals == (0,))
        assert (any(ax_val) == False)
