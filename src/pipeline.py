# Dueño: Maxi (integración). Solo librerías externas, nunca `from src.x import ...`.
# Uses build_problem (hamiltonian.py), taper_problem / build_ansatz / excitation_list /
# count_cz (ansatz.py), measure_energy / measure_reliability (measurement.py) and
# make_record (analysis.py) from the shared namespace, like every other notebook cell.
"""From Fede's optimal θ to measured energies: the reported E ± σ of every point.

The θ come from the statevector optimization (allowed by the rules); the energies
reported here are measured with shots through measure_energy (SamplerV2), as the
rules require.

    points = load_points("uccsd_tap")       # {d: {"theta", "e_exact"}} from results/uccsd_tap_d<d>.json
    records = measure_curve("uccsd_tap", points, total_shots=5_000_000)
    out = report(records)

Orbital signs: PySCF returns every molecular orbital up to an arbitrary sign, and
the sign can change between runs (it does at d = 3.0 Å, even inside one process).
Flipping orbital p flips the sign of every excitation amplitude that touches p, so
a θ optimized in one run can give a much higher energy in another. align_theta
undoes this: it tries the 2^n_orb sign patterns on θ and keeps the one whose exact
(statevector) energy reproduces the e_exact stored by the optimization. This only
fixes the parameters; the reported energy is still the one measured with shots.
"""
import glob
import itertools
import json
import os
import re
import time

import numpy as np
from qiskit.quantum_info import Statevector

_KIND_PATTERN = re.compile(r"^(uccsd|uccgsd|kupccgsd)(\d*)(_tap)?$")
ALIGN_TOLERANCE = 1e-6   # Ha


def parse_kind(name: str):
    """'uccsd2_tap' -> ('uccsd', 2, True). Inverse of the naming used by vqe.sweep."""
    m = _KIND_PATTERN.match(name)
    if not m:
        raise ValueError(f"unknown ansatz name {name!r}")
    return m.group(1), int(m.group(2) or 1), m.group(3) is not None


def load_points(name: str, results_dir: str = "results") -> dict:
    """{d: {"theta", "e_exact"}} from Fede's results/<name>_d<d>.json files."""
    points = {}
    for path in glob.glob(os.path.join(results_dir, f"{name}_d*.json")):
        with open(path) as f:
            row = json.load(f)
        if row.get("kind") == name:
            points[float(row["d"])] = {"theta": list(row["theta"]), "e_exact": float(row["e_exact"])}
    return dict(sorted(points.items()))


def prepare_point(d: float, name: str):
    """(problem, ansatz) for distance d, exactly as vqe.sweep built them."""
    kind, k, taper = parse_kind(name)
    problem = build_problem(d)
    if taper:
        problem = taper_problem(problem)
    return problem, build_ansatz(problem, kind, k)


def _exact_energy(ansatz, theta, hamiltonian):
    return float(Statevector(ansatz.assign_parameters(theta)).expectation_value(hamiltonian).real)


def align_theta(problem: dict, ansatz, name: str, theta, e_exact: float):
    """Return (θ, sign pattern) such that the exact energy of the ansatz matches e_exact.

    Each parameter belongs to one excitation (the list repeats once per rep); its
    sign changes by the product of the signs of the spatial orbitals it touches.
    Raises if no pattern reproduces e_exact, which would mean something other than
    orbital signs changed.
    """
    kind, k, _ = parse_kind(name)
    theta = np.asarray(theta, dtype=float)
    excitations = excitation_list(problem, kind, k)
    if len(theta) != k * len(excitations):
        raise ValueError(f"θ has {len(theta)} values, expected {k} x {len(excitations)} excitations")
    n_orb = problem.get("num_spatial_orbitals", problem["n_qubits"] // 2)
    touched = [[q % n_orb for q in occ + virt] for occ, virt in excitations] * k

    best = None
    for signs in itertools.product((1, -1), repeat=n_orb):
        if signs[0] == -1:
            continue   # a global flip of all orbitals changes nothing; halve the search
        factor = np.array([np.prod([signs[p] for p in orbs]) for orbs in touched])
        candidate = theta * factor
        e = _exact_energy(ansatz, candidate, problem["hamiltonian"])
        if best is None or abs(e - e_exact) < abs(best[2] - e_exact):
            best = (candidate, signs, e)
        if abs(e - e_exact) < ALIGN_TOLERANCE:
            break
    if abs(best[2] - e_exact) > ALIGN_TOLERANCE:
        raise RuntimeError(f"d={problem['d']}: no orbital sign pattern reproduces e_exact "
                           f"({best[2]:.8f} vs {e_exact:.8f})")
    return best[0], best[1]


def measure_point(d: float, name: str, point: dict, total_shots: int, seed: int = 0,
                  reliability_shots: int = 0, allocation: str = "weighted") -> dict:
    """Measure one point of the curve and return its record (see analysis.make_record).

    `point` is {"theta", "e_exact"}. With reliability_shots > 0 it also measures
    <N>, <S²> and Var(H) (theory notes, section 11), stored under "reliability".
    """
    problem, ansatz = prepare_point(d, name)
    theta, signs = align_theta(problem, ansatz, name, point["theta"], point["e_exact"])
    t0 = time.time()
    m = measure_energy(ansatz, theta, problem["hamiltonian"], total_shots,
                       allocation=allocation, seed=seed)
    extra = {"n_qubits": problem["n_qubits"], "allocation": allocation,
             "e_exact": point["e_exact"], "orbital_signs": list(signs),
             "measure_seconds": round(time.time() - t0, 2)}
    if reliability_shots > 0:
        r = measure_reliability(ansatz, theta, problem["hamiltonian"], problem["number_op"],
                                problem["s2_op"], reliability_shots, allocation, seed=seed + 1)
        extra["reliability"] = {k: v for k, v in r.items() if k not in ("energy", "energy_sigma")}
    return make_record(name, problem, theta, m, count_cz(ansatz), **extra)


def measure_curve(name: str, points: dict, total_shots: int, seed: int = 0,
                  reliability_shots: int = 0, allocation: str = "weighted",
                  verbose: bool = True) -> list:
    """measure_point for every d in `points`; seeds seed, seed+10, ... per point."""
    records = []
    for i, (d, point) in enumerate(sorted(points.items())):
        t0 = time.time()
        rec = measure_point(d, name, point, total_shots, seed + 10 * i, reliability_shots, allocation)
        records.append(rec)
        if verbose:
            err = 1e3 * abs(rec["energy"] - rec["e_fci"])
            err_ccsd = 1e3 * abs(rec["e_ccsd"] - rec["e_fci"])
            z = (rec["energy"] - rec["e_exact"]) / rec["sigma"]
            print(f"d={d:.2f}: E={rec['energy']:.6f} ± {1e3 * rec['sigma']:.2f} mHa "
                  f"(vs exacta {z:+.1f}σ), |E-FCI|={err:.2f} mHa (CCSD {err_ccsd:.2f}), "
                  f"cz={rec['cz']}, {time.time() - t0:.1f} s", flush=True)
    return records


def reliability_table(records):
    """<N>, <S²> and Var(H) of every point that was measured with reliability_shots > 0."""
    import pandas as pd
    rows = []
    for r in sorted(records, key=lambda r: r["d"]):
        rel = r.get("reliability")
        if not rel:
            continue
        rows.append({"d (Å)": r["d"],
                     "<N>": rel["N"], "σ N": rel["N_sigma"],
                     "<S²>": rel["S2"], "σ S²": rel["S2_sigma"],
                     "Var(H) (Ha²)": rel["variance"], "σ Var": rel["variance_sigma"]})
    return pd.DataFrame(rows)
