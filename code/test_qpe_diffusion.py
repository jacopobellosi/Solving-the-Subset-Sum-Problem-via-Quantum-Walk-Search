import numpy as np
from qat.lang.AQASM import Program, H, X, Z
from qat.lang.AQASM.qftarith import QFT
from qat.qpus import PyLinalg

def run_toy_amplification(use_correct_qft=True):
    """
    Simula una versione ridotta (5 qubit) del ciclo di Szegedy/Grover
    per dimostrare l'impatto del blocco compute/uncompute sul QFT.
    """
    prg = Program()
    
    # Registro di ricerca (search space), 2 qubit = 4 stati (|00>, |01>, |10>, |11>)
    search_reg = prg.qalloc(2)
    # Registro QPE (Quantum Phase Estimation), 3 qubit
    qpe_reg = prg.qalloc(3)
    
    # 1. Inizializzazione in sovrapposizione uniforme
    for q in search_reg:
        prg.apply(H, q)
    for q in qpe_reg:
        prg.apply(H, q)
        
    # --- INIZIO ITERAZIONE DI GROVER/SZEGEDY ---
    
    # 2. Oracolo: marca lo stato |11> invertendone la fase
    prg.apply(Z.ctrl(1), search_reg[0], search_reg[1])
    
    # 3. Blocco Approx Reflection (QPE + Diffusione)
    with prg.compute():
        # In un vero algoritmo qui ci sarebbe il Quantum Walk (W^2^j).
        # Per simulare l'entanglement tra lo spazio di ricerca e la QPE,
        # applichiamo dei gate controllati fittizi:
        prg.apply(Z.ctrl(1), search_reg[0], qpe_reg[0])
        prg.apply(Z.ctrl(1), search_reg[1], qpe_reg[1])

        # APPLICAZIONE DEL QFT (IL BUG RISOLTO)
        if use_correct_qft:
            prg.apply(QFT(3).dag(), qpe_reg)  # CORRETTO: Inversa per estrarre la fase
        else:
            prg.apply(QFT(3), qpe_reg)        # BUG: QFT dritto usato nel codice originale

    # 4. Diffusione sul registro QPE (Riflessione attorno allo stato |000>)
    for q in qpe_reg:
        prg.apply(X, q)
        
    prg.apply(Z.ctrl(2), qpe_reg[0], qpe_reg[1], qpe_reg[2])
    
    for q in qpe_reg:
        prg.apply(X, q)
        
    # 5. Uncompute del blocco (re-inverte tutto)
    prg.uncompute()
    # --- FINE ITERAZIONE ---

    # Creazione circuito e misurazione SOLO del registro di ricerca
    circ = prg.to_circ()
    qpu = PyLinalg()
    
    # Esegue il job quantistico
    res = qpu.submit(circ.to_job(qubits=search_reg))
    
    print("\n=======================================================")
    if use_correct_qft:
        print("RISULTATI CON IL FIX: prg.apply(QFT(len_s).dag(), qpe_s)")
    else:
        print("RISULTATI CON IL BUG: prg.apply(QFT(len_s), qpe_s)")
    print("=======================================================")
    
    for sample in res:
        # Lo stato target è |11>
        marker = " <--- TARGET (Dovrebbe amplificarsi)" if sample.state.int == 3 else ""
        print(f"Stato: {sample.state} | Probabilità: {sample.probability:.4f}{marker}")

if __name__ == "__main__":
    print("Avvio del test di isolamento QPE+Diffusione (5 qubit totali)...")
    
    # 1. Eseguiamo il circuito simulando il codice vecchio (Bug)
    run_toy_amplification(use_correct_qft=False)
    
    # 2. Eseguiamo il circuito simulando il codice corretto (Fix)
    run_toy_amplification(use_correct_qft=True)
    
    print("\nTest completato.")
