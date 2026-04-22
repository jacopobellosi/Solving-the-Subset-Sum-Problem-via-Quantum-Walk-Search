# Quantum Walk Search for the Subset-Sum Problem

Study and patched re-implementation of the MNRS quantum walk search for the Subset-Sum Problem, based on Lancellotti et al. (IEEE TC, 2026).

- `code_2025TC/` — original reference code shipped with the paper.
- `code_updated/cssp.py` — same algorithm with five documented fixes (symmetric reflections A/B, removed cleanup block, low-width insert, clamped spectral-gap parameters, inverse QFT).
- `documentation/report_modifiche/` — LaTeX report describing each fix and the root cause of the flat 0.333 output observed on the toy instance J(3,1).
- `documentation/quantum_simulation/` — chronologically numbered simulation logs, one per modification.
- `classical_comparison/` — classical baseline for cross-checking small instances.

Run: `python code_updated/cssp.py True` to execute the patched circuit (n=3, k=1, values=[1,2,3], target=3) on PyLinalg.
