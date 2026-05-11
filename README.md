# Quantum Walk Search for the Subset-Sum Problem

Study and patched re-implementation of the MNRS quantum walk search for the Constrained Subset-Sum Problem (CSSP), based on Lancellotti, Perriello, Barenghi and Pelosi, *Solving the Subset Sum Problem via Quantum Walk Search*, IEEE Transactions on Computers vol. 75, n. 1, 2026 ([DOI](https://doi.org/10.1109/TC.2025.3625044)).

## Repository layout

- `code_2025TC/` — original reference code shipped with the paper.
- `code_updated/cssp.py` — same algorithm with five documented modifications:
  1. Symmetric reflections A and B on the full coin register.
  2. Removal of the destructive α/ω cleanup block.
  3. Switch to the low-width `insert` variant inside `delete`.
  4. Clamp of the spectral gap and minimum QPE precision ℓ_s.
  5. Diagnostic checkpoints (`cssp_with_checkpoint.py`) to inspect the state at six points of one walk iteration.
- `documentation/final_report/` — final LaTeX report. Covers each modification, a scaling comparison with seven related quantum-search papers, the structured mapping against the Moguel et al. reporting guidelines, and the conclusions.
- `documentation/paper_comparison/` — the seven related papers surveyed in Section 4 of the report.
- `documentation/quantum_simulation/` — chronologically numbered simulation logs, one per modification.
- `classical_comparison/` — classical baseline for cross-checking small instances.

## Outcome

On the smallest instance J(3,1) the simulation produces a flat (1/3, 1/3, 1/3) distribution. The report shows that this is not a defect of the code: J(3,1) sits below the regime in which MNRS provides any speed-up, and the smallest non-trivial instance (n=4, k=2) already requires 38 qubits, beyond the practical reach of any state-vector simulator.

Modifications 1 and 4 were confirmed as genuine bugs by the paper authors and merged upstream as PR #1 into `paper-codes/2025-TC`, together with a rewritten `README`. The remaining modifications were kept local: Mod 2 as a local mitigation, Mod 3 as a variant choice already exposed by the existing `low_width` flag, Mod 5 as a diagnostic tool.

## Running

```
python code_updated/cssp.py True
```

Executes the patched circuit on the smallest instance (n=3, k=1, values=[1,2,3], target=3) using PyLinalg, the state-vector backend of myQLM.

## License

Released under the MIT License (see `LICENSE`). The code in `code_2025TC/` is the unmodified paper reference and remains subject to its authors' original terms.
