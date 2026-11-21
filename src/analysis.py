# Dueño: Maxi. Lee results/<kind>_d<d>.json y arma tabla, gráficos, R_e y D_e.
# Solo librerías externas, nunca `from src.x import ...`.
"""Results bookkeeping and the final analysis of the potential energy curve.

A *record* is one point of the curve, a JSON-serializable dict:
    {"kind", "d", "energy", "sigma", "shots", "cz", "theta",
     "e_hf", "e_ccsd", "e_fci", ...optional extras (allocation, reliability...)}

Pipeline:
    make_record(...) -> save_record(...)            one point -> results/<kind>_d<d>.json
    load_records(kind)                              all points of one ansatz, sorted by d
    results_table(records)                          pandas table for the notebook
    plot_curve(records), plot_errors(records)       figures
    fit_equilibrium(d, E, sigma)                    R_e and E(R_e) with bootstrap errors
    dissociation_energy(records)                    D_e = E(3.0 Å) - E(R_e)
    beats_ccsd(records)                             |E - E_FCI| + sigma < |E_CCSD - E_FCI| for d >= 1.8
    summary(records)                                all of the above for VQE, CCSD and FCI
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

RESULTS_DIR = "results"
HARTREE_TO_MHA = 1000.0
CHEMICAL_ACCURACY = 1.6e-3   # Ha
D_STRETCHED = 1.8            # Å, from here on we have to beat CCSD
D_DISSOCIATED = 3.0          # Å, atoms considered separated (D_e definition in CONSIGNA.md)


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
def make_record(kind: str, problem: dict, theta, measurement: dict, cz: int, **extra) -> dict:
    """Build a record from build_problem's dict, the optimal theta and measure_energy's result."""
    record = {
        "kind": kind,
        "d": float(problem["d"]),
        "energy": float(measurement["energy"]),
        "sigma": float(measurement["sigma"]),
        "shots": int(measurement["shots"]),
        "cz": int(cz),
        "theta": [float(t) for t in np.atleast_1d(theta)] if theta is not None else [],
        "e_hf": float(problem["e_hf"]),
        "e_ccsd": float(problem["e_ccsd"]),
        "e_fci": float(problem["e_fci"]),
    }
    record.update(_to_jsonable(extra))
    return record


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def record_path(kind: str, d: float, results_dir: str = RESULTS_DIR) -> str:
    return os.path.join(results_dir, f"{kind}_d{d:.2f}.json")


def save_record(record: dict, results_dir: str = RESULTS_DIR) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = record_path(record["kind"], record["d"], results_dir)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


def load_records(kind: str, results_dir: str = RESULTS_DIR) -> list:
    records = []
    for path in glob.glob(os.path.join(results_dir, f"{kind}_d*.json")):
        with open(path) as f:
            records.append(json.load(f))
    return sorted(records, key=lambda r: r["d"])


def _arrays(records):
    records = sorted(records, key=lambda r: r["d"])
    get = lambda key: np.array([r[key] for r in records], dtype=float)
    return records, get


# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------
def results_table(records) -> pd.DataFrame:
    """One row per distance: E ± sigma, shots, cz, references and errors in mHa."""
    rows = []
    for r in sorted(records, key=lambda r: r["d"]):
        err = abs(r["energy"] - r["e_fci"])
        err_ccsd = abs(r["e_ccsd"] - r["e_fci"])
        stretched = r["d"] >= D_STRETCHED - 1e-9
        rows.append({
            "d (Å)": r["d"],
            "E (Ha)": r["energy"],
            "σ (mHa)": r["sigma"] * HARTREE_TO_MHA,
            "shots": r["shots"],
            "cz": r["cz"],
            "E_CCSD (Ha)": r["e_ccsd"],
            "E_FCI (Ha)": r["e_fci"],
            "|E-E_FCI| (mHa)": err * HARTREE_TO_MHA,
            "|E_CCSD-E_FCI| (mHa)": err_ccsd * HARTREE_TO_MHA,
            "le gana a CCSD": bool(err + r["sigma"] < err_ccsd) if stretched else None,
            "precisión química": bool(err < CHEMICAL_ACCURACY),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_curve(records, ax=None, label="VQE"):
    """E(d) with ±sigma error bars, against HF, CCSD and FCI."""
    records, get = _arrays(records)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    d = get("d")
    ax.plot(d, get("e_hf"), ":", color="gray", label="HF")
    ax.plot(d, get("e_ccsd"), "s--", color="tab:orange", ms=4, label="CCSD")
    ax.plot(d, get("e_fci"), "-", color="black", lw=1, label="FCI")
    ax.errorbar(d, get("energy"), yerr=get("sigma"), fmt="o", color="tab:blue",
                ms=4, capsize=3, label=f"{label} (±σ)")
    ax.set_xlabel("d (Å)")
    ax.set_ylabel("E (Ha)")
    ax.set_title("H₄ lineal, STO-3G")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_errors(records, ax=None, label="VQE"):
    """|E - E_FCI| in mHa (with sigma) and the CCSD error, log scale, chemical accuracy line."""
    records, get = _arrays(records)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    d = get("d")
    err = np.abs(get("energy") - get("e_fci")) * HARTREE_TO_MHA
    sig = get("sigma") * HARTREE_TO_MHA
    err_ccsd = np.abs(get("e_ccsd") - get("e_fci")) * HARTREE_TO_MHA
    ax.plot(d, err_ccsd, "s--", color="tab:orange", ms=4, label="|E_CCSD − E_FCI|")
    ax.errorbar(d, err, yerr=sig, fmt="o", color="tab:blue", ms=4, capsize=3,
                label=f"|E_{label} − E_FCI| (±σ)")
    ax.plot(d, err + sig, "v", color="tab:blue", ms=4, alpha=0.4, label="|E − E_FCI| + σ")
    ax.axhline(CHEMICAL_ACCURACY * HARTREE_TO_MHA, color="green", ls=":", label="1,6 mHa")
    ax.axvline(D_STRETCHED, color="gray", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("d (Å)")
    ax.set_ylabel("error (mHa)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    return ax


# --------------------------------------------------------------------------
# R_e and D_e
# --------------------------------------------------------------------------
def _minimum(d, e, method, n_points):
    """Location and value of the minimum of the curve through (d, e)."""
    if method == "parabola":
        i = int(np.argmin(e))
        half = n_points // 2
        lo = max(0, min(i - half, len(d) - n_points))
        sel = slice(lo, lo + n_points)
        a, b, c = np.polyfit(d[sel], e[sel], 2)
        if a <= 0:
            return d[i], e[i]
        x = -b / (2 * a)
        return x, c - b ** 2 / (4 * a)
    if method == "spline":
        spline = CubicSpline(d, e)
        i = int(np.argmin(e))
        lo, hi = d[max(i - 1, 0)], d[min(i + 1, len(d) - 1)]
        res = minimize_scalar(spline, bounds=(lo, hi), method="bounded")
        return float(res.x), float(res.fun)
    raise ValueError("method must be 'parabola' or 'spline'")


def fit_equilibrium(d, energies, sigmas=None, method: str = "spline", n_points: int = 5,
                    n_bootstrap: int = 2000, seed: int = 0) -> dict:
    """R_e and E(R_e) from a cubic spline through all points (default), or a parabola
    through the n_points closest to the minimum.

    On the 0.1 Å grid the spline lands within ~0.3 mÅ of the dense-grid FCI minimum,
    while the parabola is off by 10 to 30 mÅ because the curve is not symmetric.

    Errors come from a parametric bootstrap: the energies are resampled as
    E_i + sigma_i * N(0, 1) and the fit is repeated. With sigmas=None (CCSD, FCI)
    the errors are zero. This is only the statistical error, the grid spacing
    also limits the precision of R_e.
    """
    d = np.asarray(d, dtype=float)
    e = np.asarray(energies, dtype=float)
    order = np.argsort(d)
    d, e = d[order], e[order]
    r_e, e_min = _minimum(d, e, method, n_points)
    r_e_sigma = e_min_sigma = 0.0
    if sigmas is not None and np.any(np.asarray(sigmas) > 0):
        s = np.asarray(sigmas, dtype=float)[order]
        rng = np.random.default_rng(seed)
        samples = np.array([_minimum(d, e + s * rng.standard_normal(len(e)), method, n_points)
                            for _ in range(n_bootstrap)])
        r_e_sigma, e_min_sigma = samples.std(axis=0, ddof=1)
    return {"R_e": float(r_e), "R_e_sigma": float(r_e_sigma),
            "E_min": float(e_min), "E_min_sigma": float(e_min_sigma), "method": method}


def dissociation_energy(records, key: str = "energy", method: str = "spline",
                        d_dissociated: float = D_DISSOCIATED, **fit_kwargs) -> dict:
    """D_e = E(3.0 Å) - E(R_e) for the curve stored under `key` ("energy", "e_ccsd" or "e_fci")."""
    records, get = _arrays(records)
    d = get("d")
    e = get(key)
    s = get("sigma") if key == "energy" else None
    j = int(np.argmin(np.abs(d - d_dissociated)))
    if abs(d[j] - d_dissociated) > 1e-6:
        raise ValueError(f"no point at d = {d_dissociated} Å")
    fit = fit_equilibrium(d, e, s, method=method, **fit_kwargs)
    s_far = 0.0 if s is None else s[j]
    return {
        **fit,
        "E_dissociated": float(e[j]),
        "D_e": float(e[j] - fit["E_min"]),
        "D_e_sigma": float(np.hypot(s_far, fit["E_min_sigma"])),
    }


def beats_ccsd(records, d_min: float = D_STRETCHED) -> dict:
    """Check |E - E_FCI| + sigma < |E_CCSD - E_FCI| on every point with d >= d_min."""
    rows = []
    for r in sorted(records, key=lambda r: r["d"]):
        if r["d"] < d_min - 1e-9:
            continue
        lhs = abs(r["energy"] - r["e_fci"]) + r["sigma"]
        rhs = abs(r["e_ccsd"] - r["e_fci"])
        rows.append({"d": r["d"], "lhs_mHa": lhs * HARTREE_TO_MHA, "rhs_mHa": rhs * HARTREE_TO_MHA,
                     "ok": bool(lhs < rhs),
                     "chemical_accuracy": bool(abs(r["energy"] - r["e_fci"]) < CHEMICAL_ACCURACY)})
    return {"all_ok": bool(rows) and all(r["ok"] for r in rows),
            "all_chemical_accuracy": bool(rows) and all(r["chemical_accuracy"] for r in rows),
            "points": rows}


def summary(records, method: str = "spline", **fit_kwargs) -> pd.DataFrame:
    """R_e and D_e for VQE (with errors), CCSD and FCI, side by side."""
    rows = []
    for name, key in (("VQE", "energy"), ("CCSD", "e_ccsd"), ("FCI", "e_fci")):
        r = dissociation_energy(records, key=key, method=method, **fit_kwargs)
        rows.append({"método": name,
                     "R_e (Å)": r["R_e"], "σ R_e (Å)": r["R_e_sigma"],
                     "E(R_e) (Ha)": r["E_min"],
                     "D_e (Ha)": r["D_e"], "σ D_e (mHa)": r["D_e_sigma"] * HARTREE_TO_MHA})
    return pd.DataFrame(rows)


def report(records, method: str = "spline"):
    """Everything the notebook shows at the end: table, figures, R_e/D_e and the CCSD check."""
    table = results_table(records)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_curve(records, axes[0])
    plot_errors(records, axes[1])
    fig.tight_layout()
    plt.close(fig)   # do not auto-display here; show it with out["figure"]
    check = beats_ccsd(records)
    return {"table": table, "summary": summary(records, method=method),
            "beats_ccsd": check, "figure": fig}
