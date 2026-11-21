# Dueño: Maxi. Solo librerías externas, nunca `from src.x import ...`.
# Uses measure_energy (measurement.py) from the shared namespace, like every notebook cell.
"""Noisy devices built from qiskit_aer's noise library, and zero-noise extrapolation.

A *device* is the dict that measure_energy(..., noise=device) understands:
    {"label", "noise_model", "basis_gates", "coupling_map", "backend", "initial_layout"}

Devices:
    make_device(n_qubits, ...)     hand-made noise, every source can be switched on alone:
        p_2q / p_1q                 depolarizing error per two-qubit (cz) / one-qubit gate  -> gate noise
        t1_us / t2_us               thermal relaxation during every gate                      -> qubit noise
        readout                     probability of reading the wrong bit                      -> qubit noise
        bad_pairs {(i, j): p}       an extra-noisy cz on one specific link
        bad_qubits {q: t1_us}       one qubit with a short T1 (T2 = T1)
        connectivity "line" | "full"
    backend_device("FakeFez", n)   the calibrated noise of a real IBM device (qiskit_ibm_runtime)

Architecture:
    two_qubit_gate_counts(circuit, device)   cz count per link after transpiling for the device

Mitigation:
    zne(circuit, theta, hamiltonian, shots, device, scales=(1, 3, 5))
        measures with every cz folded 1, 3, 5 times and extrapolates to zero noise
        (linear fit, Richardson, exponential decay towards the fully mixed energy).
"""
from collections import Counter

import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import (NoiseModel, ReadoutError, depolarizing_error,
                              thermal_relaxation_error)

BASIS_GATES = ["cz", "sx", "x", "rz"]
# Typical durations on IBM Heron (FakeFez) hardware. rz is virtual: zero time, no error.
GATE_TIME_1Q_NS = 36
GATE_TIME_2Q_NS = 68


# --------------------------------------------------------------------------
# Devices
# --------------------------------------------------------------------------
def _coupling(n_qubits, connectivity):
    if connectivity == "line":
        return CouplingMap.from_line(n_qubits)
    if connectivity == "full":
        return CouplingMap.from_full(n_qubits)
    raise ValueError("connectivity must be 'line' or 'full'")


def _relaxation(t1_us, t2_us, time_ns):
    t2_us = min(t2_us, 2 * t1_us)   # physical constraint T2 <= 2 T1
    return thermal_relaxation_error(t1_us * 1e3, t2_us * 1e3, time_ns)


def make_device(n_qubits: int, p_2q: float = 0.0, p_1q: float = 0.0,
                t1_us=None, t2_us=None, readout: float = 0.0,
                bad_pairs=None, bad_qubits=None, connectivity: str = "line",
                label: str = None) -> dict:
    """Hand-made noisy device: every error is attached per qubit / per link, so single
    qubits or links can be made worse than the rest (bad_qubits, bad_pairs)."""
    coupling = _coupling(n_qubits, connectivity)
    bad_pairs = {tuple(sorted(k)): v for k, v in (bad_pairs or {}).items()}
    bad_qubits = dict(bad_qubits or {})
    t1 = {q: bad_qubits.get(q, t1_us) for q in range(n_qubits)}
    t2 = {q: (bad_qubits[q] if q in bad_qubits else t2_us if t2_us is not None else t1_us)
          for q in range(n_qubits)}

    nm = NoiseModel(basis_gates=BASIS_GATES)
    for q in range(n_qubits):
        err = None
        if p_1q > 0:
            err = depolarizing_error(p_1q, 1)
        if t1[q] is not None:
            relax = _relaxation(t1[q], t2[q], GATE_TIME_1Q_NS)
            err = relax if err is None else err.compose(relax)
        if err is not None:
            nm.add_quantum_error(err, ["sx", "x"], [q])
        if readout > 0:
            nm.add_readout_error(ReadoutError([[1 - readout, readout], [readout, 1 - readout]]), [q])

    for i, j in sorted({tuple(sorted(e)) for e in coupling.get_edges()}):
        p = p_2q + bad_pairs.get((i, j), 0.0)
        for a, b in ((i, j), (j, i)):   # cz is symmetric, but the error is listed per qubit order
            err = depolarizing_error(min(p, 1.0), 2) if p > 0 else None
            if t1[a] is not None and t1[b] is not None:
                # A.expand(B) = B ⊗ A: A acts on the first qubit of the instruction (a)
                relax = _relaxation(t1[a], t2[a], GATE_TIME_2Q_NS).expand(
                    _relaxation(t1[b], t2[b], GATE_TIME_2Q_NS))
                err = relax if err is None else err.compose(relax)
            if err is not None:
                nm.add_quantum_error(err, "cz", [a, b])

    if label is None:
        parts = [f"p2q={p_2q:g}"] if p_2q else []
        parts += [f"p1q={p_1q:g}"] if p_1q else []
        parts += [f"T1={t1_us:g}µs"] if t1_us else []
        parts += [f"readout={readout:g}"] if readout else []
        parts += [f"bad link {k}: +{v:g}" for k, v in bad_pairs.items()]
        parts += [f"bad qubit {q}: T1={v:g}µs" for q, v in bad_qubits.items()]
        label = ", ".join(parts) or "noiseless"
        label += f" ({connectivity})"
    return {"label": label, "noise_model": nm, "basis_gates": BASIS_GATES,
            "coupling_map": coupling, "backend": None, "initial_layout": None,
            "n_qubits": n_qubits}


def backend_device(name: str = "FakeFez", n_qubits: int = None, initial_layout=None) -> dict:
    """Noise model calibrated from a real IBM device (fake backend, no account needed).

    The circuit is transpiled against the device's own coupling map and gate set;
    the transpiler picks which physical qubits to use unless initial_layout is given.
    """
    from qiskit_ibm_runtime import fake_provider
    backend = getattr(fake_provider, name)()
    return {"label": name, "noise_model": NoiseModel.from_backend(backend),
            "basis_gates": list(backend.operation_names), "coupling_map": backend.coupling_map,
            "backend": backend, "initial_layout": initial_layout, "n_qubits": n_qubits}


# --------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------
def transpile_for(circuit, device: dict):
    """The circuit as it would run on the device (same settings as measurement.py)."""
    if device.get("backend") is not None:
        return transpile(circuit, backend=device["backend"], initial_layout=device.get("initial_layout"),
                         optimization_level=1, seed_transpiler=0)
    return transpile(circuit, basis_gates=device["basis_gates"], coupling_map=device["coupling_map"],
                     initial_layout=device.get("initial_layout"), optimization_level=1, seed_transpiler=0)


def two_qubit_gate_counts(circuit, device: dict) -> Counter:
    """{(i, j): number of two-qubit gates on that link} after transpiling for the device."""
    t = transpile_for(circuit, device)
    counts = Counter()
    for inst in t.data:
        if inst.operation.num_qubits == 2 and inst.operation.name != "barrier":
            counts[tuple(sorted(t.find_bit(q).index for q in inst.qubits))] += 1
    return counts


# --------------------------------------------------------------------------
# Zero-noise extrapolation
# --------------------------------------------------------------------------
def _poly_weights(scales, degree):
    """Weights w with value(0) = sum_i w_i E_i for a least-squares polynomial fit."""
    V = np.vander(np.asarray(scales, dtype=float), degree + 1, increasing=True)
    return np.linalg.pinv(V)[0]


def extrapolate(scales, energies, sigmas, method: str = "linear", e_mixed: float = None,
                n_bootstrap: int = 2000, seed: int = 0) -> dict:
    """Value at zero noise from (scale, E ± sigma).

    linear       least-squares line through all points
    richardson   polynomial through all points (degree = n_points - 1)
    exponential  E(λ) = e_mixed + A exp(-c λ): depolarizing noise drives the state to
                 the fully mixed one, whose energy e_mixed = Tr(H) / 2^n is known
    """
    scales = np.asarray(scales, dtype=float)
    e = np.asarray(energies, dtype=float)
    s = np.asarray(sigmas, dtype=float)
    if method in ("linear", "richardson"):
        w = _poly_weights(scales, 1 if method == "linear" else len(scales) - 1)
        return {"method": method, "energy": float(w @ e), "sigma": float(np.sqrt(np.sum((w * s) ** 2)))}
    if method == "exponential":
        if e_mixed is None:
            raise ValueError("exponential extrapolation needs e_mixed = Tr(H) / 2^n")

        def fit(values):
            gap = values - e_mixed
            if np.any(gap >= 0):   # noise pushed it past the mixed energy: no decay to fit
                return np.nan
            slope, intercept = np.polyfit(scales, np.log(-gap), 1)
            return e_mixed - np.exp(intercept)

        value = fit(e)
        rng = np.random.default_rng(seed)
        boot = np.array([fit(e + s * rng.standard_normal(len(e))) for _ in range(n_bootstrap)])
        boot = boot[np.isfinite(boot)]
        return {"method": method, "energy": float(value),
                "sigma": float(boot.std(ddof=1)) if len(boot) > 1 else float("nan")}
    raise ValueError("method must be 'linear', 'richardson' or 'exponential'")


def mixed_state_energy(hamiltonian) -> float:
    """<H> in the fully mixed state = coefficient of the identity = Tr(H) / 2^n."""
    labels = hamiltonian.paulis.to_labels()
    identity = "I" * hamiltonian.num_qubits
    return float(np.real(sum(c for l, c in zip(labels, hamiltonian.coeffs) if l == identity)))


def zne(circuit, theta, hamiltonian, total_shots: int, device: dict, scales=(1, 3, 5),
        methods=("linear", "richardson", "exponential"), seed: int = 0) -> dict:
    """Measure with every cz folded `scale` times (total_shots each) and extrapolate.

    Folding multiplies only the two-qubit gate noise: one-qubit errors and readout
    errors are not scaled, so ZNE cannot remove them.
    """
    points = [measure_energy(circuit, theta, hamiltonian, total_shots, seed=seed + 7 * i,
                             noise=device, fold=k) for i, k in enumerate(scales)]
    energies = [p["energy"] for p in points]
    sigmas = [p["sigma"] for p in points]
    e_mixed = mixed_state_energy(hamiltonian)
    results = {m: extrapolate(scales, energies, sigmas, m, e_mixed=e_mixed) for m in methods}
    return {"scales": list(scales), "energies": energies, "sigmas": sigmas,
            "noisy": {"energy": energies[0], "sigma": sigmas[0]},
            "extrapolated": results, "shots": total_shots * len(scales)}
