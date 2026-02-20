# Quantum Walk Search for the Subset-Sum Problem

This repository presents a complete Qiskit implementation of a quantum walk-based search algorithm designed to solve the Subset-Sum Problem (SSP). The project is founded on the formal results presented by Lancellotti et al. (DAC '24 and IEEE TC '26) within the context of fault-tolerant quantum computation.

## Project Objective

The Subset-Sum Problem (SSP) is a well-known NP-complete problem: given a set of integers, the objective is to find a subset whose sum exactly equals a specified target value, $p$. Its applications range from resource optimization to post-quantum cryptanalysis.

While Grover's unstructured search provides a quadratic speedup over classical algorithms, this project models the search space as a Johnson Graph and explores it via a quantum walk. This structured approach yields significant asymptotic advantages by reducing standard cost metrics ($T$-count and $T$-depth) for cryptographic-grade instances, serving as a critical component for evaluating the security of future cryptographic systems.

## Algorithmic Framework

The algorithm relies on the MNRS (Magniez, Nayak, Roland, Santha) amplitude amplification framework, which alternates between marking the solution and applying an approximate reflection (diffusion) operator.

1. Search Space Representation: The solution space is modeled as a Johnson Graph $J(n, k)$, where each vertex represents a candidate subset of size $k$. Two vertices are adjacent if they differ by exactly one element.

2. QRAM-Free Setup (Vertex Binary Encoding): To bypass the necessity of physically unrealizable quantum memories (QRAM), the initial state is prepared by leveraging Dicke States to define a superposition of subset indices. Subsequently, a dedicated circuit, the Vertex Binary Encoding (VBE), dynamically decodes these indices into their actual integer values.

3. The Quantum Step (Update Operator $U_u$): To traverse the graph, the algorithm executes a structured transition: it stochastically extracts (via W-States) one element from the current subset to be discarded, replacing it with an element sampled from the remaining available values.

4. Coherence Preservation: To maintain history-independence and ensure correct quantum interference, every insertion and deletion operation keeps the data strictly ordered via a quantum insertion-sort circuit.
