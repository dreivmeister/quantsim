# =============================================================================
# profile.jl  —  profiling harness for quantsim
#
# Qubit-scaling benchmark:
#   For n qubits, builds a circuit of the form:
#     H⊗n  →  CNOT chain (n−1 gates)  →  RX⊗n (parametric)
#     → expval(PauliZ on last qubit)
#
#   Sweeps n from 1 upward, printing sim and grad timings until
#   the gradient takes > 30 s or n exceeds MAX_QUBITS.
#
#   State-vector memory: 2^n × 16 bytes (Complex128)
#   Grad evaluations:    2n full circuit runs (parameter-shift)
#
# Run with:  julia profile.jl
# =============================================================================

include("main.jl")

using Printf
using BenchmarkTools

# ─────────────────────────────────────────────────────────────────────────────
# Scaling circuit  —  H layer · CNOT chain · RX layer → expval(PauliZ)
# ─────────────────────────────────────────────────────────────────────────────

function _scaling_circuit(p, wls)
    n    = length(wls)
    ops  = (
        Tuple(H(wires=wls[i])                              for i in 1:n)...,
        Tuple(CNOT(wires=[wls[i], wls[i+1]])               for i in 1:n-1)...,
        Tuple(RX(p[i], wires=wls[i])                       for i in 1:n)...,
    )
    TypedTape(ops, (expval(PauliZ(wires=wls[n])),))
end

# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

const MAX_QUBITS    = 26
const GRAD_LIMIT_S  = 30.0   # stop when gradient exceeds this many seconds

println("─"^72)
println("Qubit scaling benchmark  (H⊗n · CNOT chain · RX⊗n → expval Z)")
println("─"^72)
@printf "%-5s  %-8s  %-8s  %-14s  %-14s\n" "n" "2^n" "MB" "sim" "grad"

for n in 1:MAX_QUBITS
    wls  = ["q$i" for i in 1:n]
    dev  = Device(wls)
    p    = rand(n)
    qn   = QNode(p -> _scaling_circuit(p, wls), dev)

    # warm-up: one call each to trigger JIT
    qn(p); grad(qn, p)

    t_sim  = @belapsed $qn($p)
    t_grad = @belapsed grad($qn, $p)
    mem_mb = (2^n * 16) / 1024^2

    sim_str  = t_sim  < 1e-3 ? @sprintf("%7.1f μs", t_sim  * 1e6) :
               t_sim  < 1.0  ? @sprintf("%7.1f ms", t_sim  * 1e3) :
                                @sprintf("%7.3f  s", t_sim)
    grad_str = t_grad < 1e-3 ? @sprintf("%7.1f μs", t_grad * 1e6) :
               t_grad < 1.0  ? @sprintf("%7.1f ms", t_grad * 1e3) :
                                @sprintf("%7.3f  s", t_grad)

    @printf "n=%-3d  %-8d  %-8.2f  %-14s  %-14s\n" n (2^n) mem_mb sim_str grad_str

    if t_grad >= GRAD_LIMIT_S
        println("  → stopped: gradient exceeded $(GRAD_LIMIT_S) s")
        break
    end
end

