import numpy as np
from scipy.optimize import minimize
from qiskit.primitives import StatevectorEstimator
import json, os


def optimize(problem: dict, ansatz: "QuantumCircuit", theta0=None, method: str = "L-BFGS-B") -> dict:
    """Minimiza <H> con statevector exacto.
    Devuelve {"theta": ndarray, "e_exact": float, "n_iters": int}.
    """
    est = StatevectorEstimator()
    H = problem["hamiltonian"]
    x0 = np.zeros(ansatz.num_parameters) if theta0 is None else np.asarray(theta0, dtype=float)

    def energy(theta):
        return float(est.run([(ansatz, H, theta)]).result()[0].data.evs)

    res = minimize(energy, x0, method=method, tol=1e-12)
    return {"theta": res.x, "e_exact": float(res.fun), "n_iters": int(res.nfev)}


def prune(problem: dict, kind: str, theta, tol: float = 1e-3) -> dict:
    """Poda saca del ansatz los saltos con |θ| <= tol y vuelve a optimizar el resto.
    Devuelve lo mismo que optimize, "ansatz" (circuito podado) y "excitations" (saltos que quedaron).
    """
    theta = np.asarray(theta, dtype=float)
    keep = np.abs(theta) > tol
    excitations = [e for e, k in zip(excitation_list(problem, kind), keep) if k]
    ansatz = build_ansatz(problem, "custom", excitations=excitations)
    return {"ansatz": ansatz, "excitations": excitations, **optimize(problem, ansatz, theta0=theta[keep])}


def sweep(kind: str, distances, out_dir: str = "results", k: int = 1, taper: bool = False) -> list:
    """Barrido de la curva con warm start (θ de cada d son el inicio del siguiente).

    kind: cualquiera de build_ansatz.
    taper: usa el problema con menos qubits (taper_problem).
    Guarda results/<kind>[k][_tap]_d<d>.json. Si ya existe, lo reutiliza.
    """
    os.makedirs(out_dir, exist_ok=True)
    name = (f"{kind}{k}" if k > 1 or kind == "kupccgsd" else kind) + ("_tap" if taper else "")
    theta, rows = None, []
    for d in sorted(distances):
        path = f"{out_dir}/{name}_d{d}.json"
        if os.path.exists(path):
            row = json.load(open(path))
        else:
            problem = build_problem(d)
            if taper:
                problem = taper_problem(problem)
            ansatz = build_ansatz(problem, kind, k)
            r = optimize(problem, ansatz, theta0=theta)
            row = {
                "d": d,
                "kind": name,
                "theta": r["theta"].tolist(),
                "e_exact": r["e_exact"],
                "n_iters": r["n_iters"],
                "cz": count_cz(ansatz),
                "n_qubits": problem["n_qubits"],
                "e_hf": problem["e_hf"],
                "e_ccsd": problem["e_ccsd"],
                "e_fci": problem["e_fci"],
            }
            json.dump(row, open(path, "w"), indent=1)
        theta = np.array(row["theta"]) if row["theta"] else None
        rows.append(row)
        print(
            f"d={d}: error {1000 * (row['e_exact'] - row['e_fci']):.4f} mHa (CCSD {1000 * (row['e_ccsd'] - row['e_fci']):.4f}), "
            f"cz {row['cz']}, parámetros {len(row['theta'])}, evaluaciones {row['n_iters']}",
            flush=True,
        )
    return rows
