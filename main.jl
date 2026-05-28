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
using StaticArrays
using BenchmarkTools
import ForwardDiff

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
# gate_matrix(g::AbstractGate)::AbstractMatrix{ComplexF64} = error("gate_matrix not implemented for $(typeof(g))")

# ── Parametric gates ──────────────────────────────────────────────────────────

rx_matrix(θ::T) where {T<:Real} = @SMatrix Complex{T}[
     cos(θ/2)      -im*sin(θ/2);
    -im*sin(θ/2)    cos(θ/2)
]

struct RXGate{T<:Real} <: AbstractGate
    θ::T
    wires::Vector{String}
    function RXGate(θ::T, wires::Vector{String}) where {T<:Real}
        length(wires) == 1 || error("RXGate requires exactly 1 wire, got $(length(wires))")
        new{T}(θ, wires)
    end
end
gate_matrix(g::RXGate{T}) where {T<:Real} = rx_matrix(g.θ)

ry_matrix(θ::T) where {T<:Real} = @SMatrix Complex{T}[
    cos(θ/2)  -sin(θ/2);
    sin(θ/2)   cos(θ/2)
]

struct RYGate{T<:Real} <: AbstractGate
    θ::T
    wires::Vector{String}
    function RYGate(θ::T, wires::Vector{String}) where {T<:Real}
        length(wires) == 1 || error("RYGate requires exactly 1 wire, got $(length(wires))")
        new{T}(θ, wires)
    end
end
gate_matrix(g::RYGate{T}) where {T<:Real} = ry_matrix(g.θ)

# ── Non-parametric gates ──────────────────────────────────────────────────────

const CNOT_MATRIX = @SMatrix ComplexF64[
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

const HADAMARD_MATRIX = @SMatrix(ComplexF64[1 1; 1 -1]) / sqrt(2)

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
shift_params(g::RXGate{T}) where {T<:Real} = (T(0.5), T(π/2))
shift_params(g::RYGate{T}) where {T<:Real} = (T(0.5), T(π/2))

# shift_gate(g, Δ)  –  return a copy of gate g with θ replaced by θ+Δ.
# Non-parametric gates have no θ; the fallback returns them unchanged.
shift_gate(g::RXGate,       Δ::Real) = RXGate(g.θ + Δ, g.wires)
shift_gate(g::RYGate,       Δ::Real) = RYGate(g.θ + Δ, g.wires)
shift_gate(g::AbstractGate, ::Real)  = g

# ─────────────────────────────────────────────────────────────────────────────
# 2. Observables
# ─────────────────────────────────────────────────────────────────────────────

const PAULI_X = @SMatrix ComplexF64[0 1; 1 0]
const PAULI_Y = @SMatrix ComplexF64[0 -im; im 0]
const PAULI_Z = @SMatrix ComplexF64[1 0; 0 -1]

struct Observable{M<:AbstractMatrix}
    matrix::M
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

struct ExpvalProcess{O<:Observable} <: AbstractMeasurement
    observable::O
end

struct VarProcess{O<:Observable} <: AbstractMeasurement
    observable::O
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

struct TypedTape{Ops<:Tuple, Meas<:Tuple}
    operations::Ops
    measurements::Meas
end

TypedTape(operations::Ops, measurements::Meas) where {Ops<:Tuple, Meas<:Tuple} =
    TypedTape{Ops, Meas}(operations, measurements)

TypedTape() = TypedTape((), ())

const Tape = TypedTape

function _replace_tuple(t::Tuple, idx::Int, value)
    return Base.setindex(t, value, idx)
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Gate and measurement functions  –  pure constructors used when building a TypedTape
# ─────────────────────────────────────────────────────────────────────────────

# Promote integer angles to Float64 for ergonomics (e.g. RX(1, wires="a")),
# but leave AbstractFloat and Dual numbers untouched so ForwardDiff can propagate.
_promote_params(p::Vector{<:AbstractFloat}) = p
_promote_params(p::Vector{<:Integer})       = float.(p)
_promote_params(p::Vector)                  = p

RX(θ::Real; wires)  = RXGate(θ isa Integer ? float(θ) : θ, _wire_vec(wires))
RY(θ::Real; wires)  = RYGate(θ isa Integer ? float(θ) : θ, _wire_vec(wires))
CNOT(; wires)       = CNOTGate(_wire_vec(wires))
H(; wires)          = HGate(_wire_vec(wires))

expval(obs::Observable) = ExpvalProcess(obs)
var(obs::Observable)    = VarProcess(obs)
probs(; wires)          = ProbsProcess(_wire_vec(wires))

# ─────────────────────────────────────────────────────────────────────────────
# 6. Device  –  statevector simulator
# ─────────────────────────────────────────────────────────────────────────────

struct Device
    wires::Vector{String}
    wire_map::Dict{String,Int}     # wire label → 1-based qubit index
end

function Device(wires::Vector{String})
    ws = collect(wires)
    Device(ws, Dict{String,Int}(w => i for (i, w) in enumerate(ws)))
end

# apply_1q_gate!
#   In-place single-qubit gate using the stride-2^(wire-1) structure of the state vector.
#   Iterates over all 2^(n-1) pairs (i0, i1) that differ only in bit `wire-1`, applying G
#   to each pair without any permutation or allocation. We only need to compute the action on the wires qubit.
#   The relevant pairs are at known offsets in the state vector, so we can compute i0 and i1 directly from the loop indices.
function apply_1q_gate!(G::AbstractMatrix{CT}, ψ::Vector{CT}, wire::Int, n::Int) where {CT<:Complex}
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

# apply_2q_gate!
#   Dedicated 2-qubit kernel: no heap allocations, no dynamic offset arrays.
#   w0 = MSB wire (gate_wires[1]), w1 = LSB wire (gate_wires[2]).
#
#   Gate matrix convention (matching gate_offsets in the generic kernel):
#     column/row r encodes the two-qubit basis state big-endian over (w0, w1):
#       r=0 (00): neither bit set  → offset 0        → i00
#       r=1 (01): w1 bit set only  → offset s1       → i01
#       r=2 (10): w0 bit set only  → offset s0       → i10
#       r=3 (11): both bits set    → offset s0+s1    → i11
#
#   Spectator enumeration: three-level loop over all base indices
#   with both bit (w0-1) = 0 and bit (w1-1) = 0, totalling 2^(n-2) bases.
#   Let s_hi = max(s0,s1), s_lo = min(s0,s1):
#     - outer iterates super-blocks of size 2·s_hi
#     - mid   iterates sub-blocks of size 2·s_lo within each s_hi half-block
#     - inner iterates individual positions within each s_lo half-block
function apply_2q_gate!(G::AbstractMatrix{CT}, ψ::Vector{CT}, w0::Int, w1::Int, n::Int) where {CT<:Complex}
    s0   = 1 << (w0 - 1)
    s1   = 1 << (w1 - 1)
    s_hi = max(s0, s1)
    s_lo = min(s0, s1)
    @inbounds for outer in 0:(2^n ÷ (2 * s_hi) - 1)
        for mid in 0:(s_hi ÷ (2 * s_lo) - 1)
            for inner in 0:(s_lo - 1)
                base = outer * 2 * s_hi + mid * 2 * s_lo + inner
                i00  = base + 1
                i01  = base + s1 + 1
                i10  = base + s0 + 1
                i11  = base + s0 + s1 + 1
                a00, a01, a10, a11 = ψ[i00], ψ[i01], ψ[i10], ψ[i11]
                ψ[i00] = G[1,1]*a00 + G[1,2]*a01 + G[1,3]*a10 + G[1,4]*a11
                ψ[i01] = G[2,1]*a00 + G[2,2]*a01 + G[2,3]*a10 + G[2,4]*a11
                ψ[i10] = G[3,1]*a00 + G[3,2]*a01 + G[3,3]*a10 + G[3,4]*a11
                ψ[i11] = G[4,1]*a00 + G[4,2]*a01 + G[4,3]*a10 + G[4,4]*a11
            end
        end
    end
end

# apply_kq_gate!
#   In-place k-qubit gate (k ≥ 3) using stride-based gather/scatter.
#   gate_wires[1] is the MSB of the gate's row/col index (big-endian convention).
#   buf is a scratch buffer of length 2^k, pre-allocated by the caller.
function apply_kq_gate!(G::AbstractMatrix{CT}, ψ::Vector{CT}, gate_wires::Vector{Int}, n::Int, buf::Vector{CT}) where {CT<:Complex}
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
            acc = zero(CT)
            for c in 0:2^k-1
                acc += G[r+1, c+1] * buf[c+1]
            end
            ψ[base + gate_offsets[r+1]] = acc
        end
    end
end

# apply_gate!
#   In-place dispatcher: 1-qubit → apply_1q_gate!, 2-qubit → apply_2q_gate!, k≥3 → apply_kq_gate!.
function apply_gate!(G::AbstractMatrix, ψ::Vector{CT}, gate_wires::Vector{Int}, n::Int, buf::Vector{CT}) where {CT<:Complex}
    G_typed = eltype(G) == CT ? G : map(CT, G)
    @assert size(G) == (2^length(gate_wires), 2^length(gate_wires))
    if length(gate_wires) == 1
        apply_1q_gate!(G_typed, ψ, gate_wires[1], n)
    elseif length(gate_wires) == 2
        apply_2q_gate!(G_typed, ψ, gate_wires[1], gate_wires[2], n)
    else
        apply_kq_gate!(G_typed, ψ, gate_wires, n, buf)
    end
end

# apply_gate  (non-mutating wrapper used by measurements)
#   Allocates a copy and delegates to apply_gate!, preserving the original ψ.
function apply_gate(G::AbstractMatrix, ψ::Vector{CT}, gate_wires::Vector{Int}, n::Int) where {CT<:Complex}
    ψ_out = copy(ψ)
    buf   = Vector{CT}(undef, 2^length(gate_wires))
    apply_gate!(G, ψ_out, gate_wires, n, buf)
    return ψ_out
end

# ⟨ψ|O|ψ⟩
function apply_measurement(meas::ExpvalProcess, ψ::Vector{CT}, wire_map::Dict{String,Int}, n::Int) where {CT<:Complex}
    wire_idx = get_wire_idxs(wire_map, meas.observable.wires)
    Oψ       = apply_gate(meas.observable.matrix, ψ, wire_idx, n)
    return real(dot(ψ, Oψ))
end

# ⟨ψ|O²|ψ⟩ - ⟨ψ|O|ψ⟩²
function apply_measurement(meas::VarProcess, ψ::Vector{CT}, wire_map::Dict{String,Int}, n::Int) where {CT<:Complex}
    wire_idx = get_wire_idxs(wire_map, meas.observable.wires)
    Oψ       = apply_gate(meas.observable.matrix, ψ, wire_idx, n)
    O2ψ      = apply_gate(meas.observable.matrix, Oψ, wire_idx, n)
    return real(dot(ψ, O2ψ)) - real(dot(ψ, Oψ))^2
end

# P(x) = |⟨x|ψ⟩|² for each computational basis state x of the selected wires.
# Iterates over all 2^n basis states, extracts the selected-wire bits (big-endian),
# and accumulates |amplitude|² into the corresponding output bin.
function apply_measurement(meas::ProbsProcess, ψ::Vector{CT}, wire_map::Dict{String,Int}, n::Int) where {CT<:Complex}
    wire_idx = get_wire_idxs(wire_map, meas.wires)
    k = length(wire_idx)
    RT = typeof(real(zero(CT)))
    probs_vec = zeros(RT, 2^k)
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

function validate_tape(tape::TypedTape, dev::Device)
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
function get_wire_idxs(wire_map::Dict{String,Int}, wires::Vector{String})
    return Int[wire_map[w] for w in wires]
end

struct QNode{F<:Function}
    func::F
    device::Device
end

# build_tape  –  execute the quantum function and require it to return a TypedTape.
#   Used by the QNode callable and by grad for pre-flight validation.
function build_tape(qn::QNode, params::Vector{<:Real})
    tape = qn.func(params)
    tape isa TypedTape || error(
        "Quantum function must return a TypedTape, got $(typeof(tape)). " *
        "Construct and return the tape directly instead of using side effects."
    )
    return tape
end

# execute_tape  –  run a pre-built tape on dev and return measurement results.
#   Assumes the tape has already been validated. Used by the QNode callable
#   and by grad to evaluate shifted tapes without re-executing qfunc.
function execute_tape(tape::TypedTape, dev::Device, ::Type{T}=Float64) where {T<:Real}
    n   = length(dev.wires)
    ψ   = zeros(Complex{T}, 2^n)
    ψ[1] = 1.0
    max_k = isempty(tape.operations) ? 1 : maximum(length(g.wires) for g in tape.operations)
    buf   = Vector{Complex{T}}(undef, 2^max_k)
    foreach(tape.operations) do gate
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
    params_f = _promote_params(params)
    tape = build_tape(qn, params_f)
    validate_tape(tape, qn.device)
    return execute_tape(tape, qn.device, eltype(params_f))
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
function grad(qn::QNode, params::Vector{<:Real})
    # ── Pre-flight validation ─────────────────────────────────────────────────
    params_f = _promote_params(params)
    T = eltype(params_f)
    tape = build_tape(qn, params_f)
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
    ops = tape.operations
    param_gate_idxs = Int[]
    for (i, gate) in enumerate(ops)
        shift_params(gate) === nothing || push!(param_gate_idxs, i)
    end

    gradient = Vector{T}(undef, length(param_gate_idxs))
    if Base.Threads.nthreads() == 1 || length(param_gate_idxs) <= 1
        for (t, i) in enumerate(param_gate_idxs)
            gate    = ops[i]
            r, s    = shift_params(gate)
            f_plus  = execute_tape(TypedTape(_replace_tuple(ops, i, shift_gate(gate,  s)), tape.measurements), dev, T)
            f_minus = execute_tape(TypedTape(_replace_tuple(ops, i, shift_gate(gate, -s)), tape.measurements), dev, T)
            gradient[t] = r * (f_plus - f_minus)
        end
    else
        Base.Threads.@threads for t in eachindex(param_gate_idxs)
            i       = param_gate_idxs[t]
            gate    = ops[i]
            r, s    = shift_params(gate)
            f_plus  = execute_tape(TypedTape(_replace_tuple(ops, i, shift_gate(gate,  s)), tape.measurements), dev, T)
            f_minus = execute_tape(TypedTape(_replace_tuple(ops, i, shift_gate(gate, -s)), tape.measurements), dev, T)
            gradient[t] = r * (f_plus - f_minus)
        end
    end
    return gradient
end

# grad_ad  –  gradient via ForwardDiff (automatic differentiation).
#
#   Computes ∂f/∂params[i] for every element of params simultaneously, using
#   dual numbers. Requires exactly one expval() measurement (same constraint as
#   grad). Returns a Vector{Float64} of length(params), one entry per input.
#
#   Use this as an independent check against the parameter-shift grad, or when
#   the circuit contains non-Pauli-rotation gates for which shift_params is not
#   defined.
function grad_ad(qn::QNode, params::Vector{<:AbstractFloat})
    # Validate with a cheap Float64 tape before paying any Dual overhead.
    tape0 = build_tape(qn, params)
    validate_tape(tape0, qn.device)
    length(tape0.measurements) == 1 || error(
        "grad_ad requires exactly one measurement, got $(length(tape0.measurements)).")
    tape0.measurements[1] isa ExpvalProcess || error(
        "grad_ad only supports expval() measurements, got $(typeof(tape0.measurements[1])).")

    return ForwardDiff.gradient(params) do p
        # eltype(p) is Dual{…} during differentiation; Dual <: Real satisfies
        # execute_tape's T<:Real constraint.  Constant gate matrices are promoted
        # lazily to Complex{Dual} via the map(CT, G) path in apply_gate!.
        execute_tape(qn.func(p), qn.device, eltype(p))
    end
end
