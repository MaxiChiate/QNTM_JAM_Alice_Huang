import numpy as np
from scipy.optimize import minimize
from qiskit.primitives import StatevectorEstimator


def optimize(problem: dict, ansatz: "QuantumCircuit", 
             theta0=None, method: str = "L-BFGS-B") -> dict:
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
