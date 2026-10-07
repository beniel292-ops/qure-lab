"""Role 1 (Quantum) configuration: every experimental constant lives here.

Noise settings are a HYPOTHETICAL SIMULATOR SWEEP. They are not derived from any
hardware calibration data, and every exported record says so.
"""

SCHEMA_VERSION = "0.1.0"          # experiment-record schema (schemas/experiment.schema.json)
INPUT_CONTRACT_VERSION = "0.1.0"  # features.csv + features_meta.json from Role 2

N_QUBITS = 4
DEFAULT_LAYERS = 2                # 4 * (layers + 1) = 12 trainable angles
BASIS_GATES = ["rz", "sx", "x", "cx"]
OBSERVABLE = "ZZZZ"               # parity of all 4 measured bits
ENCODING = "angle_i = 2*arctan(x_i)  (maps standardized features to (-pi, pi) without clipping)"

DEFAULT_SHOTS = 2000
DEFAULT_CAL_SHOTS = 2000          # per calibration circuit (2 circuits: all-0 and all-1)
DEFAULT_REPEATS = 5
BASE_SEED = 20261006

# p1q: depolarizing on 1-qubit gates sx, x (rz is virtual/noiseless in this model)
# p2q: depolarizing on cx
# readout_p10: P(read 1 | prepared 0); readout_p01: P(read 0 | prepared 1); same for every qubit
NOISE_SETTINGS = [
    {"id": "N0", "label": "noiseless", "p1q": 0.0, "p2q": 0.0, "readout_p10": 0.0, "readout_p01": 0.0},
    {"id": "N1", "label": "readout only", "p1q": 0.0, "p2q": 0.0, "readout_p10": 0.02, "readout_p01": 0.04},
    {"id": "N2", "label": "gate only", "p1q": 0.001, "p2q": 0.01, "readout_p10": 0.0, "readout_p01": 0.0},
    {"id": "N3", "label": "gate + readout (moderate)", "p1q": 0.001, "p2q": 0.01, "readout_p10": 0.02, "readout_p01": 0.04},
    {"id": "N4", "label": "gate + readout (high)", "p1q": 0.003, "p2q": 0.03, "readout_p10": 0.05, "readout_p01": 0.08},
]
NOISE_SOURCE = "hypothetical simulator sweep (not derived from hardware calibration data)"

MITIGATION = {
    "method": "tensored readout-error mitigation (per-qubit 2x2 calibration matrices, inverted)",
    "targets": "measurement (readout) errors only; does NOT address gate or decoherence errors",
    "assumption": "readout errors are independent across qubits (true for this simulator model)",
    "nonphysical_policy": "mitigated scores are reported RAW, even outside [0, 1]; never clipped",
    "calibration_prep_caveat": "the X gates used to prepare |1111> are themselves noisy under gate-noise "
                               "settings, so preparation error is absorbed into the calibration",
}


def setting_by_id(sid: str) -> dict:
    for s in NOISE_SETTINGS:
        if s["id"] == sid:
            return s
    raise KeyError(f"unknown noise setting {sid}")


def run_seed(setting_index: int, repeat: int, base: int = BASE_SEED) -> int:
    """Deterministic, recorded seed for every (setting, repeat) pair."""
    return base + 1000 * setting_index + repeat
