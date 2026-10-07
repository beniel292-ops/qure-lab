"""Noisy execution, readout calibration, mitigation and state diagnostics (Qiskit Aer).

For every case and noise setting this produces:
  noiseless_exact      exact noiseless raw score (reference, not a hardware result)
  noisy_exact          infinite-shot noisy score: gate noise via density matrix + readout applied
                       analytically. It separates "what the noise does" from shot noise.
  noisy                finite-shot noisy scores, one per repeat (seeded)
  noisy_equal_budget   finite-shot noisy scores using shots + 2*cal_shots (the same total shot budget
                       as mitigation), so raw-vs-mitigated is compared fairly
  mitigated            tensored readout-mitigated scores per repeat, RAW (may lie outside [0, 1])
  state_diagnostics    per-qubit Bloch vectors (noiseless and gate-noisy density matrix). These are
                       state diagnostics, NOT reliability or confidence measures.
"""
from __future__ import annotations

import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Pauli, Statevector, partial_trace
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from .config import BASIS_GATES, N_QUBITS, NOISE_SETTINGS, run_seed
from .model import ODD, bind, encode_angles, exact_scores, template

# ----------------------------------------------------------------------------- noise model


def gate_noise_model(s: dict) -> NoiseModel:
    nm = NoiseModel(basis_gates=BASIS_GATES)
    if s["p1q"] > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(s["p1q"], 1), ["sx", "x"])
    if s["p2q"] > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(s["p2q"], 2), ["cx"])
    return nm


def full_noise_model(s: dict) -> NoiseModel:
    nm = gate_noise_model(s)
    a, b = s["readout_p10"], s["readout_p01"]
    if a > 0 or b > 0:
        nm.add_all_qubit_readout_error(ReadoutError([[1 - a, a], [b, 1 - b]]))
    return nm


def readout_matrix(a: float, b: float) -> np.ndarray:
    """Column = prepared state, row = measured: p_meas = M @ p_true. a = P(1|0), b = P(0|1)."""
    return np.array([[1 - a, b], [a, 1 - b]])


def tensored(mats: list) -> np.ndarray:
    """Full 16x16 matrix for qubit matrices [M0, M1, M2, M3] in Qiskit little-endian order."""
    out = np.array([[1.0]])
    for m in reversed(mats):  # kron(M3, M2, M1, M0)
        out = np.kron(out, m)
    return out

# ----------------------------------------------------------------------------- helpers


def counts_to_probs(counts: dict) -> np.ndarray:
    p = np.zeros(2 ** N_QUBITS)
    tot = sum(counts.values())
    for k, v in counts.items():
        p[int(k.replace(" ", ""), 2)] += v
    return p / tot


def score_from_probs(p: np.ndarray) -> float:
    return float(p[ODD].sum())


def bloch_vectors(rho: DensityMatrix) -> list:
    out = []
    for q in range(N_QUBITS):
        red = partial_trace(rho, [i for i in range(N_QUBITS) if i != q])
        out.append([round(float(red.expectation_value(Pauli(c)).real), 6) for c in "XYZ"])
    return out


def _calibration_circuits() -> list:
    zero = QuantumCircuit(N_QUBITS)
    zero.measure_all()
    one = QuantumCircuit(N_QUBITS)
    one.x(range(N_QUBITS))
    one.measure_all()
    return [zero, one]


def calibrate(sim: AerSimulator, cal_shots: int, seed: int) -> dict:
    """Per-qubit readout calibration from |0000> and |1111> preparations."""
    circs = transpile(_calibration_circuits(), basis_gates=BASIS_GATES, optimization_level=0)
    res = sim.run(circs, shots=cal_shots, seed_simulator=seed).result()
    p0, p1 = counts_to_probs(res.get_counts(0)), counts_to_probs(res.get_counts(1))
    a, b = [], []
    for q in range(N_QUBITS):
        bit = np.array([(k >> q) & 1 for k in range(2 ** N_QUBITS)])
        a.append(float(p0[bit == 1].sum()))  # read 1 although prepared 0
        b.append(float(p1[bit == 0].sum()))  # read 0 although prepared 1
    mats = [readout_matrix(a[q], b[q]) for q in range(N_QUBITS)]
    for q, m in enumerate(mats):
        if abs(np.linalg.det(m)) < 1e-6:
            raise ValueError(f"readout calibration matrix for qubit {q} is singular")
    inv = tensored([np.linalg.inv(m) for m in mats])
    return {"a_p10": a, "b_p01": b, "inverse": inv, "shots_used": 2 * cal_shots}


def mitigate(p_meas: np.ndarray, cal: dict) -> tuple[float, bool, bool]:
    """Return (raw mitigated score, has_negative_quasiprobability, score_outside_[0,1]). NOT clipped.

    Inverting the calibration matrix gives a quasi-probability vector. With finite shots some
    entries can be slightly negative even when the parity score itself stays inside [0, 1].
    Both conditions are reported separately and nothing is clipped.
    """
    q = cal["inverse"] @ p_meas
    s = float(q[ODD].sum())
    return s, bool(np.any(q < -1e-12)), bool(s < 0 or s > 1)

# ----------------------------------------------------------------------------- main runner


def run_cases(X: np.ndarray, theta, layers: int, entangle: bool, *, settings=None, shots: int = 2000,
              cal_shots: int = 2000, repeats: int = 5, base_seed: int | None = None,
              with_state_diagnostics: bool = True, encoding: str = "arctan", log=print) -> dict:
    """Run every case under every setting. Returns {setting_id: {...per-case arrays...}}."""
    settings = settings or NOISE_SETTINGS
    theta = np.asarray(theta, dtype=float)
    angles = encode_angles(X, encoding)
    n = X.shape[0]
    s_exact = exact_scores(X, theta, layers, entangle, encoding)

    meas_t, xp, thp = template(layers, entangle, True)
    meas_t = transpile(meas_t, basis_gates=BASIS_GATES, optimization_level=0)
    unm_t, xu, thu = template(layers, entangle, False)
    unm_t = transpile(unm_t, basis_gates=BASIS_GATES, optimization_level=0)

    def bound(tmpl, xs, ths, i):
        return tmpl.assign_parameters({**dict(zip(xs, angles[i])), **dict(zip(ths, theta))})

    meas_circs = [bound(meas_t, xp, thp, i) for i in range(n)]

    diag_noiseless = None
    if with_state_diagnostics:
        diag_noiseless = [bloch_vectors(DensityMatrix(Statevector(bind(layers, entangle, False, angles[i], theta))))
                          for i in range(n)]

    out = {}
    for si, s in enumerate(NOISE_SETTINGS):
        if s not in settings:
            continue
        t0 = time.time()
        # infinite-shot noisy reference + gate-noisy state diagnostics
        dm_sim = AerSimulator(method="density_matrix", noise_model=gate_noise_model(s))
        dm_circs = []
        for i in range(n):
            c = bound(unm_t, xu, thu, i)
            c.save_density_matrix()
            dm_circs.append(c)
        dm_res = dm_sim.run(dm_circs).result()
        M = tensored([readout_matrix(s["readout_p10"], s["readout_p01"])] * N_QUBITS)
        noisy_exact, diag_noisy = np.zeros(n), []
        for i in range(n):
            rho = DensityMatrix(dm_res.data(i)["density_matrix"])
            probs = np.clip(np.real(np.diag(rho.data)), 0, None)  # remove ~1e-17 negatives only
            noisy_exact[i] = score_from_probs(M @ (probs / probs.sum()))
            if with_state_diagnostics:
                diag_noisy.append(bloch_vectors(rho))

        sim = AerSimulator(noise_model=full_noise_model(s))
        noisy = np.zeros((repeats, n))
        noisy_eq = np.zeros((repeats, n))
        mitig = np.zeros((repeats, n))
        negq = np.zeros((repeats, n), dtype=bool)
        oor = np.zeros((repeats, n), dtype=bool)
        seeds, cals = [], []
        for r in range(repeats):
            sd = run_seed(si, r, base_seed) if base_seed is not None else run_seed(si, r)
            seeds.append({"execution": sd, "calibration": sd + 500, "equal_budget": sd + 700})
            res = sim.run(meas_circs, shots=shots, seed_simulator=sd).result()
            cal = calibrate(sim, cal_shots, sd + 500)
            cals.append({"a_p10": cal["a_p10"], "b_p01": cal["b_p01"]})
            res_eq = sim.run(meas_circs, shots=shots + 2 * cal_shots, seed_simulator=sd + 700).result()
            for i in range(n):
                p = counts_to_probs(res.get_counts(i))
                noisy[r, i] = score_from_probs(p)
                mitig[r, i], negq[r, i], oor[r, i] = mitigate(p, cal)
                noisy_eq[r, i] = score_from_probs(counts_to_probs(res_eq.get_counts(i)))
        out[s["id"]] = {
            "setting": s, "noiseless_exact": s_exact, "noisy_exact": noisy_exact,
            "noisy": noisy, "noisy_equal_budget": noisy_eq, "mitigated": mitig, "negative_quasiprob": negq, "out_of_range": oor,
            "seeds": seeds, "calibrations": cals,
            "state_noiseless": diag_noiseless, "state_noisy_gate": diag_noisy if with_state_diagnostics else None,
            "runtime_s": round(time.time() - t0, 2),
            "budget": {"shots_per_execution": shots, "calibration_shots_total": 2 * cal_shots,
                       "mitigated_total_shots": shots + 2 * cal_shots,
                       "equal_budget_raw_shots": shots + 2 * cal_shots,
                       "circuit_executions_per_case_per_repeat": {"raw": 1, "mitigated": 1,
                                                                   "shared_calibration_circuits": 2}},
        }
        log(f"{s['id']} ({s['label']}): {n} cases x {repeats} repeats in {out[s['id']]['runtime_s']} s")
    return out
