from qat.lang.AQASM import Program, H, CNOT
from qat.qpus import PyLinalg

prog = Program()
q = prog.qalloc(4)
prog.apply(H, q[0])
prog.apply(CNOT, q[0], q[2])
circ = prog.to_circ()
qpu = PyLinalg()
job = circ.to_job()
res = qpu.submit(job)

for sample in res:
    state_str = str(sample.state)[1:-1]
    print(f"Prob: {sample.probability} | Ampl: {sample.amplitude} | Raw State: {sample.state} | state_str: {state_str}")
