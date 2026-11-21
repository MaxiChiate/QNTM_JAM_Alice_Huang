# Owner: Gonza. External libraries only, never `from src.x import ...` (see docs/task_split.md).
"""Chemistry of the linear H4 chain (STO-3G): from the distance d to the qubit Hamiltonian.

Agreed interface (docs/task_split.md):

    build_problem(d) -> dict with the Hamiltonian, the HF bitstring, the reference
    energies (HF, CCSD, FCI), the CCSD t1/t2 amplitudes and the N and S^2 operators.

Conventions (the same ones used by the Quantum Chemistry workshop and by qiskit-nature
with the Jordan-Wigner mapper):
  - 4 spatial orbitals -> 8 spin-orbitals -> 8 qubits.
  - Qubits 0..3 = alpha spin-orbitals, qubits 4..7 = beta spin-orbitals.
  - Molecular orbitals are sorted by Hartree-Fock energy.
  - The Hamiltonian ALREADY includes the nuclear repulsion.
"""
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo, cc
from qiskit.quantum_info import SparsePauliOp

N_HYDROGENS = 4

# Grid in Å: denser near the minimum (~0.9-1.0 Å) and in the stretched region >= 1.8 Å.
# It includes exactly 1.8 and 3.0.
DISTANCES = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0]


# --------------------------------------------------------------------------
# Jordan-Wigner
# --------------------------------------------------------------------------
def _annihilation(p, n_qubits):
    """Annihilation operator of spin-orbital p (Jordan-Wigner)."""
    return SparsePauliOp.from_sparse_list(
        [("Z" * p + "X", list(range(p + 1)), 0.5),
         ("Z" * p + "Y", list(range(p + 1)), 0.5j)],
        n_qubits,
    )


def _qubit_hamiltonian(h1, h2, constant, n_orb):
    """H = constant + sum h_pq a†_p a_q + 1/2 sum g_pqrs a†_p a†_r a_s a_q (chemists' notation)."""
    n_qubits = 2 * n_orb
    a = [_annihilation(p, n_qubits) for p in range(n_qubits)]
    a_dag = [op.adjoint() for op in a]

    # hop[s][p][q] moves an electron with spin s from orbital q to orbital p
    hop = [[[a_dag[s * n_orb + p].dot(a[s * n_orb + q]).simplify()
             for q in range(n_orb)] for p in range(n_orb)] for s in range(2)]

    terms = [SparsePauliOp("I" * n_qubits, constant)]
    for s in range(2):
        for p in range(n_orb):
            for q in range(n_orb):
                terms.append(h1[p, q] * hop[s][p][q])

    for s in range(2):
        for t in range(2):
            for p in range(n_orb):
                for q in range(n_orb):
                    second_hop = SparsePauliOp.sum(
                        [h2[p, q, r, u] * hop[t][r][u] for r in range(n_orb) for u in range(n_orb)])
                    terms.append(0.5 * hop[s][p][q].dot(second_hop))
                    if s == t:
                        # correction for the operator ordering (same spin)
                        for u in range(n_orb):
                            terms.append(-0.5 * h2[p, q, q, u] * hop[s][p][u])
    return SparsePauliOp.sum(terms).simplify(atol=1e-10)


def number_operator(n_orb):
    """N = sum of n_p = (I - Z_p)/2 over the 2*n_orb qubits."""
    n = 2 * n_orb
    terms = [SparsePauliOp("I" * n, 0.5 * n)]
    for p in range(n):
        terms.append(SparsePauliOp.from_sparse_list([("Z", [p], -0.5)], n))
    return SparsePauliOp.sum(terms).simplify()


def s2_operator(n_orb):
    """S^2 = S- S+ + Sz (Sz + 1), with alpha on qubits 0..n_orb-1 and beta on the next ones."""
    n = 2 * n_orb
    a = [_annihilation(p, n) for p in range(n)]
    a_dag = [op.adjoint() for op in a]
    s_plus = SparsePauliOp.sum([a_dag[p].dot(a[n_orb + p]) for p in range(n_orb)])
    s_minus = s_plus.adjoint()
    sz = 0.5 * SparsePauliOp.sum(
        [a_dag[p].dot(a[p]) - a_dag[n_orb + p].dot(a[n_orb + p]) for p in range(n_orb)])
    return (s_minus.dot(s_plus) + sz.dot(sz) + sz).simplify(atol=1e-10)


# --------------------------------------------------------------------------
# Full problem
# --------------------------------------------------------------------------
def geometry(d, n_h=N_HYDROGENS):
    """Straight chain of n_h hydrogens separated by d (Å), along the z axis."""
    return "; ".join(f"H 0 0 {i * d:.10f}" for i in range(n_h))


def build_problem(d: float) -> dict:
    """Linear H4, STO-3G, separation d (Å)."""
    mol = gto.M(atom=geometry(d), basis="sto-3g", verbose=0)
    hf = scf.RHF(mol)
    hf.conv_tol = 1e-12
    hf.run()
    if not hf.converged:
        raise RuntimeError(f"Hartree-Fock did not converge for d={d}")

    n_orb = mol.nao_nr()
    n_elec = mol.nelectron
    n_alpha = n_beta = n_elec // 2

    # Integrals in the molecular-orbital basis (all orbitals active)
    active = mcscf.CASCI(hf, n_orb, n_elec)
    h1, constant = active.get_h1cas()           # constant includes the nuclear repulsion
    h2 = ao2mo.restore(1, active.get_h2cas(), n_orb)
    e_fci = float(active.kernel()[0])

    ccsd = cc.CCSD(hf)
    ccsd.conv_tol = 1e-10
    ccsd.run()
    if not ccsd.converged:
        raise RuntimeError(f"CCSD did not converge for d={d}")

    # Hartree-Fock state: qubits 0..n_alpha-1 (alpha) and n_orb..n_orb+n_beta-1 (beta) occupied.
    # Qiskit convention: qubit 0 is the rightmost character.
    occupied = [1 if (q < n_alpha or n_orb <= q < n_orb + n_beta) else 0 for q in range(2 * n_orb)]
    hf_bitstring = "".join(str(b) for b in occupied[::-1])

    return {
        "d": d,
        "hamiltonian": _qubit_hamiltonian(h1, h2, constant, n_orb),
        "n_qubits": 2 * n_orb,
        "hf_bitstring": hf_bitstring,
        "e_hf": float(hf.e_tot),
        "e_ccsd": float(ccsd.e_tot),
        "e_fci": e_fci,
        "ccsd_t1": np.array(ccsd.t1),   # (n_occupied, n_virtual), spatial orbitals
        "ccsd_t2": np.array(ccsd.t2),   # (o, o, v, v), spatial orbitals
        "number_op": number_operator(n_orb),
        "s2_op": s2_operator(n_orb),
    }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def _hf_state(problem):
    """HF state as a statevector (a single computational basis state)."""
    v = np.zeros(2 ** problem["n_qubits"], dtype=complex)
    v[int(problem["hf_bitstring"], 2)] = 1.0
    return v


def validate_problem(problem: dict, penalty: float = 10.0) -> dict:
    """Lowest eigenvalue of H restricted to N=4 and S^2=0, compared with PySCF's FCI.

    A penalty on (N - 4)^2 and on S^2 is added to isolate the singlet 4-electron sector.
    Returns the error |E_min - e_fci| and the expectation values of N and S^2 in that state.
    """
    H = problem["hamiltonian"].to_matrix()
    N = problem["number_op"].to_matrix()
    S2 = problem["s2_op"].to_matrix()
    n_elec = sum(int(b) for b in problem["hf_bitstring"])
    identity = np.eye(H.shape[0])
    penalized = H + penalty * (N - n_elec * identity) @ (N - n_elec * identity) + penalty * S2
    _, vectors = np.linalg.eigh(penalized)
    psi = vectors[:, 0]
    e_min = float(np.real(psi.conj() @ H @ psi))
    hf = _hf_state(problem)
    return {
        "e_min": e_min,
        "error": abs(e_min - problem["e_fci"]),
        "N": float(np.real(psi.conj() @ N @ psi)),
        "S2": float(np.real(psi.conj() @ S2 @ psi)),
        "e_hf_check": float(np.real(hf.conj() @ H @ hf)),
    }


if __name__ == "__main__":
    for d in (1.0, 1.8, 3.0):
        p = build_problem(d)
        r = validate_problem(p)
        print(f"d={d}: E_min={r['e_min']:.10f} FCI={p['e_fci']:.10f} "
              f"|diff|={r['error']:.2e} N={r['N']:.6f} S2={r['S2']:.2e} "
              f"E(HF qubits)={r['e_hf_check']:.8f} E(HF PySCF)={p['e_hf']:.8f}")
