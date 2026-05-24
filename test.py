import pennylane as qml
import numpy as np
import time

dev = qml.device("default.qubit", wires=["a", "b"])

@qml.qnode(dev)
def circuit(params):
    qml.H(wires="a")
    qml.RX(params[0], wires="b")
    qml.CNOT(wires=["a", "b"])
    qml.RY(params[1], wires="a")
    return qml.expval(qml.PauliZ(wires="b"))

print("=== Example Circuit (PennyLane) ===\n")

# Warm up: first call triggers tracing/compilation.
circuit(np.array([0.0, 0.0]))

params = np.array([0.5, 0.3])
t0 = time.perf_counter(); result = circuit(params); t = time.perf_counter() - t0
print(f"params = {params.tolist()} -> <Z> on 'b'  =  {result}  ({t*1e6:.2f} μs)")

# Sanity check 1: zero rotation → all gates identity on b → ⟨Z_b⟩ = +1
t0 = time.perf_counter(); r0 = circuit(np.array([0.0, 0.0])); t = time.perf_counter() - t0
print(f"params = [0, 0] -> <Z> = {r0}  ({t*1e6:.2f} μs)")

# Sanity check 2: RX(π) on b flips qubit b → ⟨Z_b⟩ = -1
t0 = time.perf_counter(); rpi = circuit(np.array([np.pi, 0.0])); t = time.perf_counter() - t0
print(f"params = [π, 0] -> <Z> = {rpi}  ({t*1e6:.2f} μs)")