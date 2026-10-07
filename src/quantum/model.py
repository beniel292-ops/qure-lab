"""Four-qubit variational classifier (Qiskit).

Circuit = angle encoding RY(2*arctan(x_i)) on qubit i
        + `layers` blocks of [RY(theta) on every qubit, then CX chain 0-1, 1-2, 2-3]
        + a final RY(theta) layer.
Output  = parity of all four measured bits.  raw_score = P(odd parity) = (1 - <ZZZZ>)/2.

raw_score is a measurement-derived number in [0, 1]. It is NOT a calibrated probability
of the label. The decision threshold is chosen on validation data only.

The entangling-gate ablation (entangle=False) keeps the same parameters and observable
but removes every CX, so the circuit stays a product state.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector

from .config import BASIS_GATES, N_QUBITS, OBSERVABLE

PARITY_SIGN = np.array([(-1) ** bin(k).count("1") for k in range(2 ** N_QUBITS)], dtype=float)
ODD = PARITY_SIGN < 0


def n_params(layers: int) -> int:
    return N_QUBITS * (layers + 1)


ENCODINGS = ("arctan", "precomputed_angles")


def encode_angles(x: np.ndarray, encoding: str = "arctan") -> np.ndarray:
    """Inputs -> rotation angles.

    arctan             : standardized features -> 2*arctan(x) in (-pi, pi); smooth, monotonic, no clipping.
    precomputed_angles : inputs already ARE angles (Role 2 handoff: theta = pi*sigmoid(z) in (0, pi)); used as-is.
    """
    x = np.asarray(x, dtype=float)
    if encoding == "arctan":
        return 2.0 * np.arctan(x)
    if encoding == "precomputed_angles":
        return x
    raise ValueError(f"unknown encoding {encoding!r}; expected one of {ENCODINGS}")


def ansatz(layers: int, entangle: bool = True) -> tuple[QuantumCircuit, ParameterVector]:
    th = ParameterVector("theta", n_params(layers))
    qc = QuantumCircuit(N_QUBITS, name="ansatz")
    k = 0
    for _ in range(layers):
        for q in range(N_QUBITS):
            qc.ry(th[k], q)
            k += 1
        if entangle:
            for q in range(N_QUBITS - 1):
                qc.cx(q, q + 1)
    for q in range(N_QUBITS):
        qc.ry(th[k], q)
        k += 1
    return qc, th


@lru_cache(maxsize=8)
def template(layers: int, entangle: bool, measured: bool):
    """Parameterized full circuit (encoding + ansatz [+ measure_all]) and its parameters."""
    x = ParameterVector("x", N_QUBITS)
    qc = QuantumCircuit(N_QUBITS)
    for q in range(N_QUBITS):
        qc.ry(x[q], q)  # x here holds the ENCODED angle
    a, th = ansatz(layers, entangle)
    qc.compose(a, inplace=True)
    if measured:
        qc.measure_all()
    return qc, x, th


def bind(layers: int, entangle: bool, measured: bool, angles, theta) -> QuantumCircuit:
    qc, x, th = template(layers, entangle, measured)
    mapping = {**dict(zip(x, map(float, angles))), **dict(zip(th, map(float, theta)))}
    return qc.assign_parameters(mapping)


def encoded_states(angles: np.ndarray) -> np.ndarray:
    """Product states after RY encoding, Qiskit little-endian ordering. Shape (n, 16)."""
    c, s = np.cos(angles / 2), np.sin(angles / 2)
    out = np.empty((angles.shape[0], 2 ** N_QUBITS))
    for i in range(angles.shape[0]):
        v = np.array([1.0])
        for q in range(N_QUBITS - 1, -1, -1):  # kron(q3, q2, q1, q0)
            v = np.kron(v, np.array([c[i, q], s[i, q]]))
        out[i] = v
    return out


def exact_scores(X: np.ndarray, theta: np.ndarray, layers: int, entangle: bool = True,
                 encoding: str = "arctan") -> np.ndarray:
    """Noiseless exact raw scores for many samples at once (fast path used in training).

    Uses Qiskit's Operator for the ansatz unitary; tests check it equals Statevector of the
    full Qiskit circuit to 1e-10.
    """
    a, th = ansatz(layers, entangle)
    U = Operator(a.assign_parameters(dict(zip(th, map(float, theta))))).data
    psi = encoded_states(encode_angles(X, encoding)) @ U.T
    return (np.abs(psi) ** 2)[:, ODD].sum(axis=1)


def statevector_score(x: np.ndarray, theta: np.ndarray, layers: int, entangle: bool = True,
                      encoding: str = "arctan") -> float:
    """Reference (slow) path: full Qiskit circuit -> Statevector -> <ZZZZ>."""
    qc = bind(layers, entangle, False, encode_angles(np.atleast_2d(x), encoding)[0], theta)
    ev = Statevector(qc).expectation_value(SparsePauliOp(OBSERVABLE)).real
    return float((1 - ev) / 2)


def circuit_info(layers: int, entangle: bool) -> dict:
    qc = bind(layers, entangle, True, np.zeros(N_QUBITS), np.zeros(n_params(layers)))
    t = transpile(qc, basis_gates=BASIS_GATES, optimization_level=0, seed_transpiler=0)
    ops = {k: int(v) for k, v in t.count_ops().items()}
    return {
        "n_qubits": N_QUBITS, "layers": layers, "entangling": entangle,
        "n_trainable_params": n_params(layers),
        "logical_depth": qc.depth(), "transpiled_depth": t.depth(),
        "transpiled_ops": ops, "basis_gates": BASIS_GATES,
        "observable": OBSERVABLE, "output": "raw_score = P(odd parity of 4 measured bits)",
        "text_drawing": str(bind(layers, entangle, True, [0.1] * 4, [0.2] * n_params(layers)).draw("text")),
    }
