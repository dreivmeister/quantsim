# =============================================================================
# tests.jl  —  quantsim vs PennyLane: 10 correctness cases + speed benchmark
#
# Test cases
#   TC1  : RX(π/4)           expval(PauliZ)            — simulation + analytical
#   TC2  : RY(π/3)           expval(PauliZ)            — simulation + analytical
#   TC3  : H                  expval(PauliZ) = 0        — simulation + analytical
#   TC4  : H                  expval(PauliX) = 1        — simulation + analytical
#   TC5  : RX(π/3)            var(PauliZ) = sin²(π/3)  — simulation + analytical
#   TC6  : Bell (H+CNOT)      probs([a,b])              — simulation vector
#   TC7  : H+RX+CNOT+RY       expval(PauliZ on b)      — simulation, params=[0.5,0.3]
#   TC8  : RX(π/4)            ∂expval(Z)/∂θ            — gradient + analytical
#   TC9  : H+RX+CNOT+RY       ∇expval(Z on b)          — gradient, 2 params
#   TC10 : RX+RY+RX           ∇expval(Z)               — gradient, 3 params
#
# Run with:  julia tests.jl
# =============================================================================

ENV["JULIA_PYTHONCALL_EXE"] = "/usr/bin/python3"

using Test
using Printf
using BenchmarkTools
using PythonCall

include("main.jl")

# ─────────────────────────────────────────────────────────────────────────────
# PennyLane reference circuits  (all defined in a shared Python namespace)
# ─────────────────────────────────────────────────────────────────────────────
const PY = pydict()

pyexec("""
import pennylane as qml
import pennylane.numpy as pnp
import time

dev1 = qml.device("default.qubit", wires=["a"])
dev2 = qml.device("default.qubit", wires=["a", "b"])

# ── simulation circuits ───────────────────────────────────────────────────────

@qml.qnode(dev1, interface="autograd", diff_method="parameter-shift")
def pl_rx_z(params):
    qml.RX(params[0], wires="a")
    return qml.expval(qml.PauliZ(wires="a"))

@qml.qnode(dev1, interface="autograd", diff_method="parameter-shift")
def pl_ry_z(params):
    qml.RY(params[0], wires="a")
    return qml.expval(qml.PauliZ(wires="a"))

@qml.qnode(dev1)
def pl_h_z(params):
    qml.H(wires="a")
    return qml.expval(qml.PauliZ(wires="a"))

@qml.qnode(dev1)
def pl_h_x(params):
    qml.H(wires="a")
    return qml.expval(qml.PauliX(wires="a"))

@qml.qnode(dev1)
def pl_rx_var_z(params):
    qml.RX(params[0], wires="a")
    return qml.var(qml.PauliZ(wires="a"))

@qml.qnode(dev2)
def pl_bell_probs(params):
    qml.H(wires="a")
    qml.CNOT(wires=["a", "b"])
    return qml.probs(wires=["a", "b"])

@qml.qnode(dev2, interface="autograd", diff_method="parameter-shift")
def pl_main_circuit(params):
    qml.H(wires="a")
    qml.RX(params[0], wires="b")
    qml.CNOT(wires=["a", "b"])
    qml.RY(params[1], wires="a")
    return qml.expval(qml.PauliZ(wires="b"))

@qml.qnode(dev1, interface="autograd", diff_method="parameter-shift")
def pl_rxryrx(params):
    qml.RX(params[0], wires="a")
    qml.RY(params[1], wires="a")
    qml.RX(params[2], wires="a")
    return qml.expval(qml.PauliZ(wires="a"))

# ── gradient helpers (wrap requires_grad boilerplate) ────────────────────────
_grad_rx_z   = qml.grad(pl_rx_z)
_grad_main   = qml.grad(pl_main_circuit)
_grad_rxryrx = qml.grad(pl_rxryrx)

def grad_rx_z(theta):
    return _grad_rx_z(pnp.array([theta], requires_grad=True)).tolist()

def grad_main(t1, t2):
    return _grad_main(pnp.array([t1, t2], requires_grad=True)).tolist()

def grad_rxryrx(t1, t2, t3):
    return _grad_rxryrx(pnp.array([t1, t2, t3], requires_grad=True)).tolist()

# ── benchmark helpers ─────────────────────────────────────────────────────────
def bench_circuit(n=500):
    p = pnp.array([0.5, 0.3])
    for _ in range(10): pl_main_circuit(p)        # warm-up
    t0 = time.perf_counter()
    for _ in range(n): pl_main_circuit(p)
    return (time.perf_counter() - t0) / n * 1e9   # ns per call

def bench_grad(n=500):
    p = pnp.array([0.5, 0.3], requires_grad=True)
    for _ in range(10): _grad_main(p)             # warm-up
    t0 = time.perf_counter()
    for _ in range(n): _grad_main(p)
    return (time.perf_counter() - t0) / n * 1e9
""", PY)

# ─────────────────────────────────────────────────────────────────────────────
# PythonCall convenience wrappers
# ─────────────────────────────────────────────────────────────────────────────
const pnp = pyimport("pennylane.numpy")

py_f64(x)  = pyconvert(Float64,          pybuiltins.float(x))
py_vec(x)  = pyconvert(Vector{Float64},  x)

# ─────────────────────────────────────────────────────────────────────────────
# Julia (quantsim) circuit definitions
# ─────────────────────────────────────────────────────────────────────────────
function _jl_rx_z(p)
    TypedTape((RX(p[1], wires="a"),), (expval(PauliZ(wires="a")),))
end

function _jl_ry_z(p)
    TypedTape((RY(p[1], wires="a"),), (expval(PauliZ(wires="a")),))
end

function _jl_h_z(p)
    TypedTape((H(wires="a"),), (expval(PauliZ(wires="a")),))
end

function _jl_h_x(p)
    TypedTape((H(wires="a"),), (expval(PauliX(wires="a")),))
end

function _jl_rx_var(p)
    TypedTape((RX(p[1], wires="a"),), (var(PauliZ(wires="a")),))
end

function _jl_bell(p)
    TypedTape((
        H(wires="a"),
        CNOT(wires=["a","b"]),
    ), (
        probs(wires=["a","b"]),
    ))
end

function _jl_main(p)
    TypedTape((
        H(wires="a"),
        RX(p[1], wires="b"),
        CNOT(wires=["a","b"]),
        RY(p[2], wires="a"),
    ), (
        expval(PauliZ(wires="b")),
    ))
end

function _jl_rxryrx(p)
    TypedTape((
        RX(p[1], wires="a"),
        RY(p[2], wires="a"),
        RX(p[3], wires="a"),
    ), (
        expval(PauliZ(wires="a")),
    ))
end

const qn_rx     = QNode(_jl_rx_z,   Device(["a"]))
const qn_ry     = QNode(_jl_ry_z,   Device(["a"]))
const qn_h_z    = QNode(_jl_h_z,    Device(["a"]))
const qn_h_x    = QNode(_jl_h_x,    Device(["a"]))
const qn_rx_var = QNode(_jl_rx_var, Device(["a"]))
const qn_bell   = QNode(_jl_bell,   Device(["a","b"]))
const qn_main   = QNode(_jl_main,   Device(["a","b"]))
const qn_rxryrx = QNode(_jl_rxryrx, Device(["a"]))

# Warm-up all circuits to avoid cold-start latency in benchmarks
for _ in 1:3
    qn_rx([0.5]);          qn_ry([0.5])
    qn_h_z([0.0]);         qn_h_x([0.0])
    qn_rx_var([0.5]);      qn_bell([0.0])
    qn_main([0.5, 0.3]);   qn_rxryrx([0.1, 0.2, 0.3])
    grad(qn_rx,     [π/4])
    grad(qn_main,   [0.5, 0.3])
    grad(qn_rxryrx, [0.1, 0.2, 0.3])
end

# =============================================================================
# 10 correctness test cases
# =============================================================================
const ATOL = 1e-6

@testset "quantsim vs PennyLane – 10 test cases" begin

    # ── TC1: RX(π/4) expval(PauliZ) ─────────────────────────────────────────
    # Analytical: ⟨0|RX†(θ) Z RX(θ)|0⟩ = cos(θ)
    @testset "TC1: RX(π/4) · expval(PauliZ)" begin
        θ  = π/4
        jl = qn_rx([θ])
        pl = py_f64(PY["pl_rx_z"](pnp.array([θ])))
        @test jl ≈ pl      atol=ATOL
        @test jl ≈ cos(θ)  atol=ATOL
    end

    # ── TC2: RY(π/3) expval(PauliZ) ─────────────────────────────────────────
    # Analytical: cos(π/3) = 0.5
    @testset "TC2: RY(π/3) · expval(PauliZ)" begin
        θ  = π/3
        jl = qn_ry([θ])
        pl = py_f64(PY["pl_ry_z"](pnp.array([θ])))
        @test jl ≈ pl      atol=ATOL
        @test jl ≈ cos(θ)  atol=ATOL
    end

    # ── TC3: H expval(PauliZ) ────────────────────────────────────────────────
    # H|0⟩ = |+⟩,  ⟨+|Z|+⟩ = 0
    @testset "TC3: H · expval(PauliZ) = 0" begin
        jl = qn_h_z([0.0])
        pl = py_f64(PY["pl_h_z"](pnp.array([0.0])))
        @test jl ≈ pl      atol=ATOL
        @test abs(jl) < ATOL
    end

    # ── TC4: H expval(PauliX) ────────────────────────────────────────────────
    # H|0⟩ = |+⟩,  ⟨+|X|+⟩ = 1
    @testset "TC4: H · expval(PauliX) = 1" begin
        jl = qn_h_x([0.0])
        pl = py_f64(PY["pl_h_x"](pnp.array([0.0])))
        @test jl ≈ pl      atol=ATOL
        @test jl ≈ 1.0     atol=ATOL
    end

    # ── TC5: RX(π/3) var(PauliZ) ─────────────────────────────────────────────
    # var(Z) = ⟨Z²⟩ − ⟨Z⟩² = 1 − cos²(θ) = sin²(θ)
    @testset "TC5: RX(π/3) · var(PauliZ) = sin²(π/3)" begin
        θ  = π/3
        jl = qn_rx_var([θ])
        pl = py_f64(PY["pl_rx_var_z"](pnp.array([θ])))
        @test jl ≈ pl        atol=ATOL
        @test jl ≈ sin(θ)^2  atol=ATOL
    end

    # ── TC6: Bell state probs ─────────────────────────────────────────────────
    # H|0⟩_a ⊗ |0⟩_b → CNOT → (|00⟩ + |11⟩)/√2
    # probs = [P(00), P(01), P(10), P(11)] = [0.5, 0, 0, 0.5]
    @testset "TC6: Bell state · probs([a,b]) = [0.5,0,0,0.5]" begin
        jl = qn_bell([0.0])
        pl = py_vec(PY["pl_bell_probs"](pnp.array([0.0])))
        @test jl ≈ pl                     atol=ATOL
        @test jl ≈ [0.5, 0.0, 0.0, 0.5]  atol=ATOL
        @test sum(jl) ≈ 1.0               atol=ATOL
    end

    # ── TC7: H + RX + CNOT + RY expval(PauliZ on b) ──────────────────────────
    # Non-trivial 2-qubit entangled circuit; no simple closed form → compare only to PL
    @testset "TC7: H+RX(0.5)+CNOT+RY(0.3) · expval(PauliZ on b)" begin
        params = [0.5, 0.3]
        jl = qn_main(params)
        pl = py_f64(PY["pl_main_circuit"](pnp.array(params)))
        @test jl ≈ pl atol=ATOL
    end

    # ── TC8: gradient ∂/∂θ [RX(θ) expval(Z)] ─────────────────────────────────
    # Analytical: d/dθ cos(θ) = −sin(θ); at θ=π/4 ≈ −0.7071
    @testset "TC8: ∂/∂θ [RX(θ) expval(Z)] = −sin(θ), θ=π/4" begin
        θ    = π/4
        jl_g = grad(qn_rx, [θ])
        pl_g = py_vec(PY["grad_rx_z"](θ))
        @test jl_g    ≈ pl_g      atol=ATOL
        @test jl_g[1] ≈ -sin(θ)  atol=ATOL
    end

    # ── TC9: gradient ∇[H+RX+CNOT+RY expval(Z on b)] ────────────────────────
    @testset "TC9: ∇[H+RX+CNOT+RY expval(Z on b)], params=[0.5,0.3]" begin
        params = [0.5, 0.3]
        jl_g  = grad(qn_main, params)
        pl_g  = py_vec(PY["grad_main"](params[1], params[2]))
        @test jl_g ≈ pl_g atol=ATOL
    end

    # ── TC10: gradient ∇[RX+RY+RX expval(Z)], 3 params ──────────────────────
    @testset "TC10: ∇[RX+RY+RX expval(Z)], params=[0.1,0.2,0.3]" begin
        params = [0.1, 0.2, 0.3]
        jl_g  = grad(qn_rxryrx, params)
        pl_g  = py_vec(PY["grad_rxryrx"](params[1], params[2], params[3]))
        @test jl_g ≈ pl_g atol=ATOL
    end

end # @testset

# =============================================================================
# Speed & memory benchmark
#   Circuit: H + RX + CNOT + RY  on  ["a","b"],  params = [0.5, 0.3]
#   Gradient: parameter-shift rule for same circuit
# =============================================================================
println()
println("─"^68)
println("Benchmark · H + RX(0.5) + CNOT + RY(0.3) circuit  ([\"a\",\"b\"])")
println("─"^68)

const _bp = [0.5, 0.3]

b_sim  = @benchmark $qn_main($_bp)           seconds=3
b_grad = @benchmark grad($qn_main, $_bp)     seconds=3

pl_ns_sim  = py_f64(PY["bench_circuit"]())
pl_ns_grad = py_f64(PY["bench_grad"]())

jl_ns_sim  = median(b_sim).time
jl_ns_grad = median(b_grad).time

println("\nSimulation:")
@printf "  Julia     (median) : %9.1f ns   allocs: %d\n"  jl_ns_sim  median(b_sim).allocs
@printf "  PennyLane (mean)   : %9.1f ns\n"               pl_ns_sim
@printf "  speedup            : %9.2f×\n"                  pl_ns_sim / jl_ns_sim

println("\nGradient (parameter-shift, 2 params):")
@printf "  Julia     (median) : %9.1f ns   allocs: %d\n"  jl_ns_grad  median(b_grad).allocs
@printf "  PennyLane (mean)   : %9.1f ns\n"               pl_ns_grad
@printf "  speedup            : %9.2f×\n"                  pl_ns_grad / jl_ns_grad

println("\nMemory  (Julia, @allocated, single call):")
@printf "  circuit  : %6d bytes\n"  @allocated qn_main([0.5, 0.3])
@printf "  gradient : %6d bytes\n"  @allocated grad(qn_main, [0.5, 0.3])
