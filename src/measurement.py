# Dueño: Maxi. Solo librerías externas, nunca `from src.x import ...`.
"""Shot-based estimation of expectation values with qiskit_aer's SamplerV2.

Public API (interface in docs/task_split.md):
    measure_energy(circuit, theta, hamiltonian, total_shots) -> {"energy", "sigma", "shots"}

Extras:
    measure_expectation   same thing for any SparsePauliOp (N, S², H², ...)
    validate_sigma        repeat the measurement with different seeds and compare
                          the spread of the energies with the reported sigma
    shots_for_sigma       shots needed to reach a target sigma (sigma ~ 1/sqrt(shots))
    compare_allocations   uniform vs |c_k|-weighted vs adaptive shot allocation
    measure_reliability   <N>, <S²> and Var(H) = <H²> - <H>², measured with shots

Noise (`noise=`, optional): a dict describing a noisy device, built by noise.py:
    {"noise_model": NoiseModel, "basis_gates": [...], "coupling_map": CouplingMap or None,
     "backend": backend to transpile against or None, "initial_layout": [...] or None}
The measurement circuits are transpiled to that device and simulated with the
noise model (density matrix). `fold=k` (odd) repeats every two-qubit gate k times
after transpiling, which multiplies its noise by ~k: this is the noise scaling
used by zero-noise extrapolation (ZNE).

Shot allocation strategies (`allocation=`):
    "uniform"   same number of shots for every qubit-wise commuting group
    "weighted"  shots proportional to sum_k |c_k| over the group
    "adaptive"  a pilot run (uniform) estimates the per-shot std sigma_g of every
                group, and the remaining shots go proportional to sigma_g, which
                minimizes sum_g sigma_g² / N_g for a fixed total (Neyman allocation)
"""
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2

ALLOCATIONS = ("uniform", "weighted", "adaptive")
MIN_SHOTS_PER_GROUP = 2   # at least 2 shots so that the sample variance is defined
PILOT_FRACTION = 0.1      # fraction of the budget spent on the pilot run in "adaptive"


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def _split_identity(operator):
    """Return (constant, operator without the identity term). The constant needs no shots."""
    operator = operator.simplify(atol=1e-12)
    labels = operator.paulis.to_labels()
    identity = "I" * operator.num_qubits
    constant = float(np.real(sum(c for l, c in zip(labels, operator.coeffs) if l == identity)))
    keep = [i for i, l in enumerate(labels) if l != identity]
    rest = operator[keep] if keep else None
    return constant, rest


def _group_basis(group):
    """Basis per qubit ('I', 'X', 'Y' or 'Z') shared by all terms of a qubit-wise commuting group."""
    n = group.num_qubits
    basis = ["I"] * n
    for label in group.paulis.to_labels():
        for q in range(n):
            p = label[n - 1 - q]  # Qiskit labels: qubit 0 is the rightmost character
            if p != "I":
                basis[q] = p
    return basis


def _measurement_circuit(state_circuit, basis):
    """Append the basis-change rotations and a measurement of every qubit."""
    qc = state_circuit.copy()
    qc.barrier()
    for q, p in enumerate(basis):
        if p == "X":
            qc.h(q)
        elif p == "Y":
            qc.sdg(q)
            qc.h(q)
    qc.measure_all()
    return qc


def _per_shot_values(counts, group):
    """Return (values, weights): value of sum_k c_k P_k for each distinct bitstring, and its count.

    A Pauli string measured in its eigenbasis gives the parity of the bits on its support:
    +1 for an even number of ones, -1 for an odd number.
    """
    coeffs = np.real(group.coeffs)
    support = group.paulis.x | group.paulis.z                 # (n_terms, n_qubits), qubit 0 first
    bitstrings = list(counts)
    # bitstring[-1] is qubit 0, so reverse to get columns in qubit order
    bits = np.array([[int(b) for b in s.replace(" ", "")[::-1]] for s in bitstrings], dtype=int)
    parity = (bits @ support.T.astype(int)) % 2                # (n_outcomes, n_terms)
    values = (1 - 2 * parity) @ coeffs
    weights = np.array([counts[s] for s in bitstrings], dtype=float)
    return values, weights


def _group_stats(counts, group):
    """Mean and per-shot sample variance of the group's observable."""
    values, weights = _per_shot_values(counts, group)
    n = weights.sum()
    mean = np.sum(weights * values) / n
    var = np.sum(weights * (values - mean) ** 2) / max(n - 1, 1)
    return mean, var, n


def _distribute(total, scores):
    """Split `total` shots proportionally to `scores` (largest remainder), at least MIN_SHOTS_PER_GROUP each."""
    k = len(scores)
    scores = np.asarray(scores, dtype=float)
    floor = MIN_SHOTS_PER_GROUP * k
    if total <= floor:
        return np.full(k, max(MIN_SHOTS_PER_GROUP, total // k), dtype=int)
    if scores.sum() <= 0:
        scores = np.ones(k)
    extra = total - floor
    raw = extra * scores / scores.sum()
    shots = np.floor(raw).astype(int)
    remainder = extra - shots.sum()
    shots[np.argsort(raw - shots)[::-1][:remainder]] += 1
    return shots + MIN_SHOTS_PER_GROUP


def fold_two_qubit_gates(circuit, fold: int):
    """Replace every two-qubit gate G by G (G† G)^((fold-1)/2). Same unitary, ~fold times the noise.

    Must be applied after the last transpile, or the optimizer cancels the pairs.
    """
    if fold == 1:
        return circuit
    if fold < 1 or fold % 2 == 0:
        raise ValueError(f"fold must be an odd integer >= 1, got {fold}")
    folded = circuit.copy_empty_like()
    for inst in circuit.data:
        folded.append(inst)
        if inst.operation.num_qubits == 2 and inst.operation.name not in ("barrier", "measure"):
            inverse = inst.operation.inverse()
            for _ in range((fold - 1) // 2):
                folded.append(inverse, inst.qubits, inst.clbits)
                folded.append(inst.operation, inst.qubits, inst.clbits)
    return folded


def _prepare_circuits(circuits, noise, fold):
    """Transpile the measurement circuits for the (noisy) device and apply the folding."""
    if noise is None:
        if fold != 1:
            raise ValueError("fold only makes sense with a noise model")
        return transpile(circuits, AerSimulator())
    transpiled = transpile(circuits, backend=noise.get("backend"),
                           basis_gates=None if noise.get("backend") else noise["basis_gates"],
                           coupling_map=None if noise.get("backend") else noise.get("coupling_map"),
                           initial_layout=noise.get("initial_layout"),
                           optimization_level=1, seed_transpiler=0)
    return [fold_two_qubit_gates(qc, fold) for qc in transpiled]


def _sampler(noise, seed):
    backend_options = {}
    if noise is not None:
        backend_options["noise_model"] = noise["noise_model"]
        backend_options["method"] = "density_matrix"   # exact noisy distribution, then sample
    if seed is not None:
        backend_options["seed_simulator"] = int(seed)
    return SamplerV2(options={"backend_options": backend_options}) if backend_options else SamplerV2()


def _run(sampler, circuits, shots):
    """One SamplerV2 job with a different number of shots per circuit; returns a list of counts."""
    pubs = [(qc, None, int(s)) for qc, s in zip(circuits, shots)]
    result = sampler.run(pubs).result()
    return [r.data.meas.get_counts() for r in result]


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------
def measure_expectation(circuit, theta, operator, total_shots: int,
                        allocation: str = "weighted", seed=None, noise=None, fold: int = 1) -> dict:
    """Estimate <operator> with `total_shots` shots spread over qubit-wise commuting groups.

    The sigma accounts for the covariances inside each group exactly, because the
    per-shot value of the whole group (sum_k c_k P_k on one bitstring) is what gets
    averaged. Different groups are independent circuits, so their variances add.

    Returns {"energy", "sigma", "shots", "n_groups", "allocation", "groups"}; "groups"
    has the shots, mean and per-shot std of every group.
    """
    if allocation not in ALLOCATIONS:
        raise ValueError(f"allocation must be one of {ALLOCATIONS}, got {allocation!r}")

    qc = circuit.assign_parameters(theta) if circuit.num_parameters > 0 else circuit
    constant, rest = _split_identity(operator)
    if rest is None:
        return {"energy": constant, "sigma": 0.0, "shots": 0, "n_groups": 0,
                "allocation": allocation, "groups": []}

    groups = rest.group_commuting(qubit_wise=True)
    circuits = _prepare_circuits([_measurement_circuit(qc, _group_basis(g)) for g in groups], noise, fold)
    # Aer seeds the i-th circuit of a job with seed_simulator + i, so nearby user seeds
    # would reuse random streams across calls. Derive well-separated job seeds instead.
    job_seeds = np.random.SeedSequence(seed).generate_state(2) if seed is not None else None

    def sampler_for(job):
        return _sampler(noise, None if job_seeds is None else job_seeds[job])

    sampler = sampler_for(0)

    if allocation == "uniform":
        shots = _distribute(total_shots, np.ones(len(groups)))
        counts = _run(sampler, circuits, shots)
    elif allocation == "weighted":
        shots = _distribute(total_shots, [np.sum(np.abs(g.coeffs)) for g in groups])
        counts = _run(sampler, circuits, shots)
    else:  # adaptive
        pilot = _distribute(int(PILOT_FRACTION * total_shots), np.ones(len(groups)))
        pilot_counts = _run(sampler, circuits, pilot)
        stds = np.array([np.sqrt(_group_stats(c, g)[1]) for c, g in zip(pilot_counts, groups)])
        # if a group gave zero spread in the pilot, keep a small weight so it is not starved
        stds = np.maximum(stds, 1e-3 * max(stds.max(), 1e-12))
        # Neyman target for the total budget, minus what each group already got in the pilot
        target = _distribute(total_shots, stds)
        main = np.maximum(target - pilot, 0)
        main_counts = _run(sampler_for(1), circuits, np.maximum(main, 1))
        counts = []
        for pc, mc, m in zip(pilot_counts, main_counts, main):
            merged = dict(pc)
            if m > 0:
                for k, v in mc.items():
                    merged[k] = merged.get(k, 0) + v
            counts.append(merged)
        shots = pilot + main

    energy = constant
    variance = 0.0
    details = []
    for group, c in zip(groups, counts):
        mean, var, n = _group_stats(c, group)
        energy += mean
        variance += var / n
        details.append({"n_terms": len(group), "shots": int(n),
                        "mean": float(mean), "std_per_shot": float(np.sqrt(var))})

    return {
        "energy": float(energy),
        "sigma": float(np.sqrt(variance)),
        "shots": int(sum(d["shots"] for d in details)),
        "n_groups": len(groups),
        "allocation": allocation,
        "fold": fold,
        "groups": details,
    }


def measure_energy(circuit, theta, hamiltonian, total_shots: int,
                   allocation: str = "weighted", seed=None, noise=None, fold: int = 1) -> dict:
    """Devuelve {"energy": float, "sigma": float, "shots": int}.

    Groups the Pauli terms into qubit-wise commuting sets, measures one circuit
    per group (shots split according to `allocation`) and returns the energy
    estimate with its statistical standard error. The identity term is added
    exactly and costs no shots. With `noise` the circuits run on a noisy simulated
    device, and `fold` scales its two-qubit noise (see the module docstring).
    """
    r = measure_expectation(circuit, theta, hamiltonian, total_shots, allocation, seed, noise, fold)
    return {"energy": r["energy"], "sigma": r["sigma"], "shots": r["shots"]}


# --------------------------------------------------------------------------
# Validation of sigma and shot budget
# --------------------------------------------------------------------------
def validate_sigma(circuit, theta, hamiltonian, total_shots: int, n_repeats: int = 30,
                   allocation: str = "weighted", seed0: int = 0) -> dict:
    """Repeat measure_energy `n_repeats` times with seeds seed0, seed0+1, ...

    If sigma is right, the standard deviation of the energies ("std_energies")
    matches the reported sigma ("mean_sigma") within the sampling error of a std
    estimated from n samples, about std / sqrt(2 (n - 1)).
    """
    runs = [measure_energy(circuit, theta, hamiltonian, total_shots, allocation, seed=seed0 + i)
            for i in range(n_repeats)]
    energies = np.array([r["energy"] for r in runs])
    sigmas = np.array([r["sigma"] for r in runs])
    std = float(np.std(energies, ddof=1))
    return {
        "allocation": allocation,
        "total_shots": int(runs[0]["shots"]),
        "n_repeats": n_repeats,
        "mean_energy": float(energies.mean()),
        "std_energies": std,
        "std_energies_error": std / np.sqrt(2 * (n_repeats - 1)),
        "mean_sigma": float(sigmas.mean()),
        "ratio": std / float(sigmas.mean()),
        "energies": energies,
        "sigmas": sigmas,
    }


def shots_for_sigma(sigma: float, shots: int, target: float = 1e-3) -> int:
    """Shots needed to reach `target` given that `shots` gave `sigma` (sigma ~ 1/sqrt(shots))."""
    return int(np.ceil(shots * (sigma / target) ** 2))


def compare_allocations(circuit, theta, hamiltonian, total_shots: int, n_repeats: int = 30,
                        target: float = 1e-3, allocations=ALLOCATIONS, seed0: int = 0) -> list:
    """For every allocation: reported sigma, empirical std over repeats, and shots for sigma < target."""
    rows = []
    for allocation in allocations:
        v = validate_sigma(circuit, theta, hamiltonian, total_shots, n_repeats, allocation, seed0)
        rows.append({
            "allocation": allocation,
            "shots": v["total_shots"],
            "mean_energy": v["mean_energy"],
            "mean_sigma": v["mean_sigma"],
            "std_energies": v["std_energies"],
            "ratio": v["ratio"],
            "shots_for_target": shots_for_sigma(v["mean_sigma"], v["total_shots"], target),
        })
    return rows


# --------------------------------------------------------------------------
# Reliability indicators (theory notes, section 11)
# --------------------------------------------------------------------------
def measure_reliability(circuit, theta, hamiltonian, number_op, s2_op, total_shots: int,
                        allocation: str = "weighted", seed=None) -> dict:
    """Measure <N>, <S²> and Var(H) with shots, using the same code as the energy.

    Var(H) = <(H - a)²> - (<H> - a)². Shifting by a = measured <H> keeps the
    coefficients of (H - a)² small, so its estimate is much less noisy than
    measuring <H²> and <H>² separately; the second term is then ~sigma_E² and
    is dropped. Var(H) = 0 if and only if the state is an eigenstate.
    """
    s = None if seed is None else int(seed)
    energy = measure_expectation(circuit, theta, hamiltonian, total_shots, allocation, s)
    a = energy["energy"]
    n_qubits = hamiltonian.num_qubits
    shifted = (hamiltonian - SparsePauliOp("I" * n_qubits, a)).simplify(atol=1e-12)
    h2 = shifted.dot(shifted).simplify(atol=1e-12)
    var = measure_expectation(circuit, theta, h2, total_shots, allocation,
                              None if s is None else s + 10)
    n = measure_expectation(circuit, theta, number_op, total_shots, allocation,
                            None if s is None else s + 20)
    s2 = measure_expectation(circuit, theta, s2_op, total_shots, allocation,
                             None if s is None else s + 30)
    return {
        "energy": a, "energy_sigma": energy["sigma"],
        "variance": var["energy"], "variance_sigma": var["sigma"],
        "N": n["energy"], "N_sigma": n["sigma"],
        "S2": s2["energy"], "S2_sigma": s2["sigma"],
        "shots": energy["shots"] + var["shots"] + n["shots"] + s2["shots"],
    }
