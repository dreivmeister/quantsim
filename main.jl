# =============================================================================
# Quantum Circuit Simulator
# =============================================================================
#
# Components
#  1. Gate matrices     – complex matrix representations of common gates
#  2. Observables       – Hermitian operators used in measurements
#  3. Gate types        – AbstractGate hierarchy; each type carries its params and implements gate_matrix
#  4. Tape              – ordered record of AbstractGate + AbstractMeasurement
#  5. Gate functions    – user-facing RX / RY / CNOT / … that queue into the active tape
#  6. Device            – statevector simulator (2^n complex amplitudes)
#  7. QNode             – binds a quantum function to a device and executes it
#  8. Gradients         – parameter-shift rule for exact derivatives of expval circuits
#
# =============================================================================

using LinearAlgebra
using BenchmarkTools

# ─────────────────────────────────────────────────────────────────────────────
# 1. Gate matrices and types  –  each carries its params and computes its matrix lazily
# ─────────────────────────────────────────────────────────────────────────────
#
# gate_matrix(g) is called at evaluation time, not at circuit-construction time.
# This means θ can later be a dual number for automatic differentiation:
# swap Float64 for a Dual type and derivatives flow through automatically.

abstract type AbstractGate end
# Interface every concrete gate must satisfy:
#   gate_matrix(g)::Matrix{ComplexF64}  – unitary matrix applied to the state
gate_matrix(g::AbstractGate)::Matrix{ComplexF64} = error("gate_matrix not implemented for $(typeof(g))")

# ── Parametric gates ──────────────────────────────────────────────────────────
# θ is Float64 now; swapping to a Dual type later enables ForwardDiff.

rx_matrix(θ::Real) = ComplexF64[
     cos(θ/2)      -im*sin(θ/2);
    -im*sin(θ/2)    cos(θ/2)
]

struct RXGate <: AbstractGate
    θ::Float64
    wires::Vector{String}
    function RXGate(θ::Float64, wires::Vector{String})
        length(wires) == 1 || error("RXGate requires exactly 1 wire, got $(length(wires))")
        new(θ, wires)
    end
end
gate_matrix(g::RXGate) = rx_matrix(g.θ)

ry_matrix(θ::Real) = ComplexF64[
    cos(θ/2)  -sin(θ/2);
    sin(θ/2)   cos(θ/2)
]

struct RYGate <: AbstractGate
    θ::Float64
    wires::Vector{String}
    function RYGate(θ::Float64, wires::Vector{String})
        length(wires) == 1 || error("RYGate requires exactly 1 wire, got $(length(wires))")
        new(θ, wires)
    end
end
gate_matrix(g::RYGate) = ry_matrix(g.θ)

# ── Non-parametric gates ──────────────────────────────────────────────────────

const CNOT_MATRIX = ComplexF64[
    1 0 0 0;
    0 1 0 0;
    0 0 0 1;
    0 0 1 0
]

struct CNOTGate <: AbstractGate
    wires::Vector{String}
    function CNOTGate(wires::Vector{String})
        length(wires) == 2 || error("CNOTGate requires exactly 2 wires, got $(length(wires))")
        new(wires)
    end
end
gate_matrix(::CNOTGate) = CNOT_MATRIX

const HADAMARD_MATRIX = ComplexF64[1 1; 1 -1] / sqrt(2)

struct HGate <: AbstractGate
    wires::Vector{String}
    function HGate(wires::Vector{String})
        length(wires) == 1 || error("HGate requires exactly 1 wire, got $(length(wires))")
        new(wires)
    end
end
gate_matrix(::HGate) = HADAMARD_MATRIX

# ── Parameter-shift data ─────────────────────────────────────────────────────
# shift_params(g) returns (r, s) such that  ∂f/∂θ = r · [f(θ+s) − f(θ−s)].
# Returns nothing for gates that have no free parameter.
# For Pauli rotation gates  U(θ) = exp(−iθ/2 · P)  the exact values are r=1/2, s=π/2.
shift_params(::AbstractGate) = nothing
shift_params(::RXGate)       = (0.5, π/2)
shift_params(::RYGate)       = (0.5, π/2)

# shift_gate(g, Δ)  –  return a copy of gate g with θ replaced by θ+Δ.
# Non-parametric gates have no θ; the fallback returns them unchanged.
shift_gate(g::RXGate,       Δ::Real) = RXGate(g.θ + Δ, g.wires)
shift_gate(g::RYGate,       Δ::Real) = RYGate(g.θ + Δ, g.wires)
shift_gate(g::AbstractGate, ::Real)  = g

# ─────────────────────────────────────────────────────────────────────────────
# 2. Observables
# ─────────────────────────────────────────────────────────────────────────────

const PAULI_X = ComplexF64[0 1; 1 0]
const PAULI_Y = ComplexF64[0 -im; im 0]
const PAULI_Z = ComplexF64[1 0; 0 -1]

struct Observable
    matrix::Matrix{ComplexF64}
    wires::Vector{String}
end

# Helper to allow wires="a" or wires=["a","b"] syntax in gate and observable constructors.
_wire_vec(w::AbstractString)      = String[w]
_wire_vec(w::AbstractVector)      = convert(Vector{String}, w)

PauliX(; wires) = Observable(PAULI_X, _wire_vec(wires))
PauliY(; wires) = Observable(PAULI_Y, _wire_vec(wires))
PauliZ(; wires) = Observable(PAULI_Z, _wire_vec(wires))

# ─────────────────────────────────────────────────────────────────────────────
# 3. Measurements
# ─────────────────────────────────────────────────────────────────────────────

abstract type AbstractMeasurement end

struct ExpvalProcess <: AbstractMeasurement
    observable::Observable
end

struct VarProcess <: AbstractMeasurement
    observable::Observable
end

# probs: stores which wires to compute marginal probabilities over
struct ProbsProcess <: AbstractMeasurement
    wires::Vector{String}
end

# Wire accessor — dispatch so validate_tape doesn't need to know the internal layout
measurement_wires(m::ExpvalProcess) = m.observable.wires
measurement_wires(m::VarProcess)    = m.observable.wires
measurement_wires(m::ProbsProcess)  = m.wires

# ─────────────────────────────────────────────────────────────────────────────
# 4. Tape  –  ordered record of gates + measurements
# ─────────────────────────────────────────────────────────────────────────────

mutable struct Tape
    operations::Vector{AbstractGate}
    measurements::Vector{AbstractMeasurement}
end

Tape() = Tape(AbstractGate[], AbstractMeasurement[])

# ─────────────────────────────────────────────────────────────────────────────
# 5. Gate and measurement functions  –  called inside a quantum function, push structs onto the active tape
# ─────────────────────────────────────────────────────────────────────────────

RX(θ::Real; wires)  = push!(task_local_storage(:active_tape).operations, RXGate(Float64(θ),  _wire_vec(wires)))
RY(θ::Real; wires)  = push!(task_local_storage(:active_tape).operations, RYGate(Float64(θ),  _wire_vec(wires)))
CNOT(; wires)       = push!(task_local_storage(:active_tape).operations, CNOTGate(_wire_vec(wires)))
H(; wires)          = push!(task_local_storage(:active_tape).operations, HGate(_wire_vec(wires)))

# expval / var / probs: record the measurement and return nothing (QNode collects results).
function expval(obs::Observable)
    push!(task_local_storage(:active_tape).measurements, ExpvalProcess(obs))
    return nothing
end

function var(obs::Observable)
    push!(task_local_storage(:active_tape).measurements, VarProcess(obs))
    return nothing
end

function probs(; wires)
    push!(task_local_storage(:active_tape).measurements, ProbsProcess(_wire_vec(wires)))
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Device  –  statevector simulator
# ─────────────────────────────────────────────────────────────────────────────

struct Device
    wires::Vector{String}
    wire_map::Dict{String,Int}     # wire label → 1-based qubit index
end

function Device(wires::AbstractVector{String})
    ws = collect(wires)
    Device(ws, Dict{String,Int}(w => i for (i, w) in enumerate(ws)))
end

# apply_1q_gate!
#   In-place single-qubit gate using the stride-2^(wire-1) structure of the state vector.
#   Iterates over all 2^(n-1) pairs (i0, i1) that differ only in bit `wire-1`, applying G
#   to each pair without any permutation or allocation. We only need to compute the action on the wires qubit.
#   The relevant pairs are at known offsets in the state vector, so we can compute i0 and i1 directly from the loop indices.
function apply_1q_gate!(G::Matrix{ComplexF64}, ψ::Vector{ComplexF64}, wire::Int, n::Int)
    s = 1 << (wire - 1)      # stride = 2^(wire-1)
    @inbounds for outer in 0:(2^n ÷ (2s) - 1)
        for inner in 0:(s - 1)
            i0 = outer * 2s + inner + 1   # 1-based index: wire bit = 0
            i1 = i0 + s                    #               wire bit = 1
            a, b = ψ[i0], ψ[i1]
            ψ[i0] = G[1,1]*a + G[1,2]*b
            ψ[i1] = G[2,1]*a + G[2,2]*b
        end
    end
end

# apply_kq_gate!
#   In-place k-qubit gate using stride-based gather/scatter.
#   gate_wires[1] is the MSB of the gate's row/col index (big-endian convention).
#   buf is a scratch buffer of length 2^k, pre-allocated by the caller.
#   Here the gate acts on k qubits, so we gather 2^k amplitudes into the buffer, apply G, and scatter back.
function apply_kq_gate!(G::Matrix{ComplexF64}, ψ::Vector{ComplexF64}, gate_wires::Vector{Int}, n::Int, buf::Vector{ComplexF64})
    k = length(gate_wires)
    # offset in ψ for gate-subspace basis state r (0-based, big-endian across gate_wires)
    gate_offsets = [sum(((r >> (k-j)) & 1) * (1 << (gate_wires[j]-1)) for j in 1:k)
                    for r in 0:2^k-1]

    spec_wires   = [w for w in 1:n if w ∉ gate_wires]
    spec_strides = [1 << (w-1) for w in spec_wires]

    @inbounds for s in 0:2^(n-k)-1
        # 1-based base index from the spectator-qubit configuration
        base = sum(((s >> (j-1)) & 1) * spec_strides[j] for j in 1:length(spec_wires); init=0) + 1

        # gather 2^k amplitudes into scratch buffer
        for r in 0:2^k-1
            buf[r+1] = ψ[base + gate_offsets[r+1]]
        end

        # apply G and scatter back
        for r in 0:2^k-1
            acc = zero(ComplexF64)
            for c in 0:2^k-1
                acc += G[r+1, c+1] * buf[c+1]
            end
            ψ[base + gate_offsets[r+1]] = acc
        end
    end
end

# apply_gate!
#   In-place dispatcher: routes to the 1-qubit specialisation or the general k-qubit path.
function apply_gate!(G::Matrix{ComplexF64}, ψ::Vector{ComplexF64}, gate_wires::Vector{Int}, n::Int, buf::Vector{ComplexF64})
    @assert size(G) == (2^length(gate_wires), 2^length(gate_wires))
    if length(gate_wires) == 1
        apply_1q_gate!(G, ψ, gate_wires[1], n)
    else
        apply_kq_gate!(G, ψ, gate_wires, n, buf)
    end
end

# apply_gate  (non-mutating wrapper used by measurements)
#   Allocates a copy and delegates to apply_gate!, preserving the original ψ.
function apply_gate(G::Matrix{ComplexF64}, ψ::Vector{ComplexF64}, gate_wires::Vector{Int}, n::Int)
    ψ_out = copy(ψ)
    buf   = Vector{ComplexF64}(undef, 2^length(gate_wires))
    apply_gate!(G, ψ_out, gate_wires, n, buf)
    return ψ_out
end

# ⟨ψ|O|ψ⟩
function apply_measurement(meas::ExpvalProcess, ψ::Vector{ComplexF64}, wire_map::AbstractDict, n::Int)
    wire_idx = get_wire_idxs(wire_map, meas.observable.wires)
    Oψ       = apply_gate(meas.observable.matrix, ψ, wire_idx, n)
    return real(dot(ψ, Oψ))
end

# ⟨ψ|O²|ψ⟩ - ⟨ψ|O|ψ⟩²
function apply_measurement(meas::VarProcess, ψ::Vector{ComplexF64}, wire_map::AbstractDict, n::Int)
    wire_idx = get_wire_idxs(wire_map, meas.observable.wires)
    Oψ       = apply_gate(meas.observable.matrix, ψ, wire_idx, n)
    O2ψ      = apply_gate(meas.observable.matrix, Oψ, wire_idx, n)
    return real(dot(ψ, O2ψ)) - real(dot(ψ, Oψ))^2
end

# P(x) = |⟨x|ψ⟩|² for each computational basis state x of the selected wires.
# Iterates over all 2^n basis states, extracts the selected-wire bits (big-endian),
# and accumulates |amplitude|² into the corresponding output bin.
function apply_measurement(meas::ProbsProcess, ψ::Vector{ComplexF64}, wire_map::AbstractDict, n::Int)
    wire_idx = get_wire_idxs(wire_map, meas.wires)
    k = length(wire_idx)
    probs_vec = zeros(Float64, 2^k)
    @inbounds for i in 0:2^n-1
        r = 0
        for (j, w) in enumerate(wire_idx)
            r |= ((i >> (w-1)) & 1) << (k-j)   # big-endian: wire_idx[1] = MSB
        end
        probs_vec[r+1] += abs2(ψ[i+1])
    end
    return probs_vec
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. QNode  –  binds a quantum function to a device
# ─────────────────────────────────────────────────────────────────────────────

function validate_tape(tape::Tape, dev::Device)
    for gate in tape.operations
        for w in gate.wires
            haskey(dev.wire_map, w) || error(
                "Gate $(nameof(typeof(gate))) references wire \"$w\" which is not in the device. " *
                "Available wires: $(dev.wires)"
            )
        end
    end
    for meas in tape.measurements
        for w in measurement_wires(meas)
            haskey(dev.wire_map, w) || error(
                "Measurement references wire \"$w\" which is not in the device. " *
                "Available wires: $(dev.wires)"
            )
        end
    end
end

# Convert wire labels to their corresponding indices in the state vector.
function get_wire_idxs(wire_map::AbstractDict, wires::AbstractVector)
    return [wire_map[w] for w in wires]
end

struct QNode
    func::Function
    device::Device
end

# build_tape  –  dry-run: execute the quantum function to populate a Tape
#   without touching the statevector simulator.  Used by the QNode callable
#   and by grad for pre-flight validation.
function build_tape(qn::QNode, params::Vector{Float64})
    tape = Tape()
    task_local_storage(:active_tape, tape) do
        qn.func(params)
    end
    return tape
end

# execute_tape  –  run a pre-built tape on dev and return measurement results.
#   Assumes the tape has already been validated. Used by the QNode callable
#   and by grad to evaluate shifted tapes without re-executing qfunc.
function execute_tape(tape::Tape, dev::Device)
    n   = length(dev.wires)
    ψ   = zeros(ComplexF64, 2^n)
    ψ[1] = 1.0
    max_k = isempty(tape.operations) ? 1 : maximum(length(g.wires) for g in tape.operations)
    buf   = Vector{ComplexF64}(undef, 2^max_k)
    for gate in tape.operations
        wire_idx = get_wire_idxs(dev.wire_map, gate.wires)
        apply_gate!(gate_matrix(gate), ψ, wire_idx, n, buf)
    end
    results = map(tape.measurements) do meas
        apply_measurement(meas, ψ, dev.wire_map, n)
    end
    return length(results) == 1 ? only(results) : results
end

# Make QNode callable: circuit([p1, p2, …])
function (qn::QNode)(params::Vector{<:Real})
    tape = build_tape(qn, Float64.(params))
    validate_tape(tape, qn.device)
    return execute_tape(tape, qn.device)
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. Gradients  –  parameter-shift rule
# ─────────────────────────────────────────────────────────────────────────────

# grad  –  exact gradient via the parameter-shift rule, applied per gate.
#
#   For each parametric gate g_i with shift_params(g_i) = (r, s):
#     ∂f/∂θ_{g_i} = r · [f(tape with θ_{g_i}+s) − f(tape with θ_{g_i}−s)]
#
#   Only gate g_i's θ is shifted; every other gate is held fixed. This correctly
#   respects each gate's own (r, s) and isolates individual gate contributions.
#
#   Returns a Vector{Float64} with one entry per parametric gate in tape order.
#   For circuits where each params[i] feeds exactly one gate, the result is
#   identical to differentiating w.r.t. params[i].
#
# Constraints (checked before differentiation):
#   · Exactly one measurement, and it must be an expval() (scalar output).
#   · Every gate whose struct carries a θ field must have shift_params defined.
function grad(qn::QNode, params::Vector{Float64})
    # ── Pre-flight validation ─────────────────────────────────────────────────
    tape = build_tape(qn, params)
    dev  = qn.device

    length(tape.measurements) == 1 || error(
        "grad requires exactly one measurement, got $(length(tape.measurements)). " *
        "Only circuits with a single expval() are supported."
    )
    tape.measurements[1] isa ExpvalProcess || error(
        "grad only supports expval() measurements, got $(typeof(tape.measurements[1]))."
    )
    for gate in tape.operations
        if hasproperty(gate, :θ) && shift_params(gate) === nothing
            error(
                "Gate $(nameof(typeof(gate))) has a parameter θ but no shift_params defined. " *
                "Add a shift_params dispatch for this gate type to make it differentiable."
            )
        end
    end
    validate_tape(tape, dev)

    # ── Per-gate parameter-shift ──────────────────────────────────────────────
    # ops is a shallow copy; gate structs are immutable so ops[i] = ... never
    # aliases back into tape.operations.
    ops      = copy(tape.operations)
    gradient = Float64[]
    for (i, gate) in enumerate(tape.operations)
        sp = shift_params(gate)
        sp === nothing && continue
        r, s    = sp
        ops[i]  = shift_gate(gate,  s)
        f_plus  = execute_tape(Tape(ops, tape.measurements), dev)
        ops[i]  = shift_gate(gate, -s)
        f_minus = execute_tape(Tape(ops, tape.measurements), dev)
        ops[i]  = gate                  # restore
        push!(gradient, r * (f_plus - f_minus))
    end
    return gradient
end








# =============================================================================
# Example circuit
#
#   Wires:  "a" (qubit 1, MSB),  "b" (qubit 2, LSB)
#   Initial state:  |00⟩
#
#   RX(params[1])  on  "b"     → rotates qubit b around X by params[1]
#   CNOT           on  ["a","b"] → flips b conditioned on a
#   RY(params[2])  on  "a"     → rotates qubit a around Y by params[2]
#   measure ⟨Z⟩ on "b"
#
# Analytical result (for |00⟩ start):
#   Because qubit a starts in |0⟩, CNOT does nothing to the state, so
#   the two qubits remain a product state throughout.
#   ⟨Z_b⟩ = cos(params[1])   (independent of params[2])
# =============================================================================


# Define the quantum function that builds the circuit on the tape when executed.
function qfunc(params)
    H(wires="a")
    RX(params[1], wires="b")
    CNOT(wires=["a", "b"])
    RY(params[2], wires="a")
    return expval(PauliZ(wires="b"))
end

dev     = Device(["a", "b"])
circuit = QNode(qfunc, dev)

println("=== Example Circuit ===\n")

# Warm up
circuit([0.0, 0.0])

params = [0.5, 0.3]
r1 = @btime circuit($params)
println("  └─ params = $params  ->  ⟨Z⟩ = $r1\n")

# Sanity check 1: zero rotation → all gates identity → state stays |00⟩ → ⟨Z_b⟩ = +1
p0 = [0.0, 0.0]
r0 = @btime circuit($p0)
println("  └─ params = $p0  ->  ⟨Z⟩ = $r0\n")

# Sanity check 2: RX(π) on b flips qubit b → state |01⟩ → ⟨Z_b⟩ = -1
ppi = [π, 0.0]
rpi = @btime circuit($ppi)
println("  └─ params = [π, 0]  ->  ⟨Z⟩ = $rpi\n")

# =============================================================================
# Gradient verification
#
#   Single-qubit circuit:  RX(θ) on "a", measure ⟨Z_a⟩
#   Analytical:  ⟨Z⟩ = cos(θ)   →   d⟨Z⟩/dθ = −sin(θ)
#
#   Two-qubit circuit  (same as above, params=[θ1, θ2]):
#   H on "a", RX(θ1) on "b", CNOT ["a","b"], RY(θ2) on "a", measure ⟨Z_b⟩
#   ∂/∂θ1:  re-run with shifted θ1  (RX is the only θ1-dependent gate)
#   ∂/∂θ2:  re-run with shifted θ2  (RY is the only θ2-dependent gate)
# =============================================================================

println("=== Gradient Verification ===\n")

# ── 1-qubit reference circuit ─────────────────────────────────────────────────
function qfunc_1q(params)
    RX(params[1], wires="a")
    return expval(PauliZ(wires="a"))
end

dev_1q     = Device(["a"])
circuit_1q = QNode(qfunc_1q, dev_1q)
circuit_1q([0.0])   # warm up

θ_test = π / 4
g_ps   = @btime grad(circuit_1q, [θ_test])
g_fd   = [(circuit_1q([θ_test + 1e-5]) - circuit_1q([θ_test - 1e-5])) / 2e-5]
g_ana  = [-sin(θ_test)]
println("  1-qubit RX, θ = π/4")
println("    parameter-shift : $(round.(g_ps;  digits=8))")
println("    finite-diff     : $(round.(g_fd;  digits=8))")
println("    analytical      : $(round.(g_ana; digits=8))\n")

#── 2-qubit circuit (reuse the circuit defined above) ─────────────────────────
p_grad = [0.5, 0.3]
g2_ps  = grad(circuit, p_grad)
g2_fd  = [(circuit([p_grad[1]+1e-5, p_grad[2]]) - circuit([p_grad[1]-1e-5, p_grad[2]])) / 2e-5,
          (circuit([p_grad[1], p_grad[2]+1e-5]) - circuit([p_grad[1], p_grad[2]-1e-5])) / 2e-5]
println("  2-qubit H+RX+CNOT+RY circuit, params = $p_grad")
println("    parameter-shift : $(round.(g2_ps; digits=8))")
println("    finite-diff     : $(round.(g2_fd; digits=8))\n")


function qfunc_grad(params)
    RX(params[1], wires="a")
    RY(params[2], wires="a")
    RX(params[3], wires="a")
    return expval(PauliZ(wires="a"))
end

dev_grad     = Device(["a"])
circuit_grad = QNode(qfunc_grad, dev_grad)

params_grad = [0.1, 0.2, 0.3]
g_grad = grad(circuit_grad, params_grad)
println("  1-qubit RX+RY+RX circuit, params = $params_grad")
println("    parameter-shift : $(round.(g_grad; digits=8))\n")