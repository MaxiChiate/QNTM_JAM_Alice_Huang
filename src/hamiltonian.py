"""Química de H4 lineal (STO-3G): de la distancia d al hamiltoniano en qubits.

Dueño: Gonza. Interfaz acordada en docs/task_split.md:

    build_problem(d) -> dict con hamiltoniano, bitstring de HF, energías de
    referencia (HF, CCSD, FCI), amplitudes t1/t2 de CCSD y los operadores N y S².

Convenciones (las mismas que usa el workshop de Química Cuántica y qiskit-nature
con el mapeo de Jordan-Wigner):
  - 4 orbitales espaciales -> 8 spin-orbitales -> 8 qubits.
  - Qubits 0..3 = spin-orbitales alfa, qubits 4..7 = spin-orbitales beta.
  - Orbital moleculares ordenados por energía de Hartree-Fock.
  - El hamiltoniano YA incluye la repulsión nuclear.
"""
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo, cc
from qiskit.quantum_info import SparsePauliOp

N_HIDROGENOS = 4

# Grilla en Å: más densa cerca del mínimo (~0,9-1,0 Å) y en la zona >= 1,8 Å.
# Incluye exactamente 1,8 y 3,0.
DISTANCES = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0]


# --------------------------------------------------------------------------
# Jordan-Wigner
# --------------------------------------------------------------------------
def _destruccion(p, n_qubits):
    """Operador de destrucción del spin-orbital p (Jordan-Wigner)."""
    return SparsePauliOp.from_sparse_list(
        [("Z" * p + "X", list(range(p + 1)), 0.5),
         ("Z" * p + "Y", list(range(p + 1)), 0.5j)],
        n_qubits,
    )


def _hamiltoniano_qubits(h, g, constante, norb):
    """H = constante + sum h_pq a†_p a_q + 1/2 sum g_pqrs a†_p a†_r a_s a_q  (notación química)."""
    n_qubits = 2 * norb
    a = [_destruccion(p, n_qubits) for p in range(n_qubits)]
    a_dag = [op.adjoint() for op in a]

    # salto[s][p][q] mueve un electrón de spin s del orbital q al orbital p
    salto = [[[a_dag[s * norb + p].dot(a[s * norb + q]).simplify()
               for q in range(norb)] for p in range(norb)] for s in range(2)]

    terminos = [SparsePauliOp("I" * n_qubits, constante)]
    for s in range(2):
        for p in range(norb):
            for q in range(norb):
                terminos.append(h[p, q] * salto[s][p][q])

    for s in range(2):
        for t in range(2):
            for p in range(norb):
                for q in range(norb):
                    segundo = SparsePauliOp.sum(
                        [g[p, q, r, u] * salto[t][r][u] for r in range(norb) for u in range(norb)])
                    terminos.append(0.5 * salto[s][p][q].dot(segundo))
                    if s == t:
                        # corrección por el orden de operadores (mismo spin)
                        for u in range(norb):
                            terminos.append(-0.5 * g[p, q, q, u] * salto[s][p][u])
    return SparsePauliOp.sum(terminos).simplify(atol=1e-10)


def operador_numero(norb):
    """N = suma de n_p = (I - Z_p)/2 sobre los 2*norb qubits."""
    n = 2 * norb
    terminos = [SparsePauliOp("I" * n, 0.5 * n)]
    for p in range(n):
        terminos.append(SparsePauliOp.from_sparse_list([("Z", [p], -0.5)], n))
    return SparsePauliOp.sum(terminos).simplify()


def operador_s2(norb):
    """S² = S- S+ + Sz (Sz + 1), con alfa en los qubits 0..norb-1 y beta en los siguientes."""
    n = 2 * norb
    a = [_destruccion(p, n) for p in range(n)]
    a_dag = [op.adjoint() for op in a]
    s_mas = SparsePauliOp.sum([a_dag[p].dot(a[norb + p]) for p in range(norb)])
    s_menos = s_mas.adjoint()
    sz = 0.5 * SparsePauliOp.sum(
        [a_dag[p].dot(a[p]) - a_dag[norb + p].dot(a[norb + p]) for p in range(norb)])
    return (s_menos.dot(s_mas) + sz.dot(sz) + sz).simplify(atol=1e-10)


# --------------------------------------------------------------------------
# Problema completo
# --------------------------------------------------------------------------
def geometria(d, n_h=N_HIDROGENOS):
    """Cadena recta de n_h hidrógenos separados por d (Å), sobre el eje z."""
    return "; ".join(f"H 0 0 {i * d:.10f}" for i in range(n_h))


def build_problem(d: float) -> dict:
    """H4 lineal, STO-3G, separación d (Å)."""
    mol = gto.M(atom=geometria(d), basis="sto-3g", verbose=0)
    hf = scf.RHF(mol)
    hf.conv_tol = 1e-12
    hf.run()
    if not hf.converged:
        raise RuntimeError(f"Hartree-Fock no convergió para d={d}")

    norb = mol.nao_nr()
    nelec = mol.nelectron
    n_alfa = n_beta = nelec // 2

    # Integrales en la base de orbitales moleculares (todos los orbitales activos)
    activo = mcscf.CASCI(hf, norb, nelec)
    h1, constante = activo.get_h1cas()          # constante incluye la repulsión nuclear
    h2 = ao2mo.restore(1, activo.get_h2cas(), norb)
    e_fci = float(activo.kernel()[0])

    ccsd = cc.CCSD(hf)
    ccsd.conv_tol = 1e-10
    ccsd.run()
    if not ccsd.converged:
        raise RuntimeError(f"CCSD no convergió para d={d}")

    # Estado de Hartree-Fock: qubits 0..n_alfa-1 (alfa) y norb..norb+n_beta-1 (beta) ocupados.
    # Convención Qiskit: el qubit 0 es el carácter de más a la derecha.
    ocupados = [1 if (q < n_alfa or norb <= q < norb + n_beta) else 0 for q in range(2 * norb)]
    hf_bitstring = "".join(str(b) for b in ocupados[::-1])

    return {
        "d": d,
        "hamiltonian": _hamiltoniano_qubits(h1, h2, constante, norb),
        "n_qubits": 2 * norb,
        "hf_bitstring": hf_bitstring,
        "e_hf": float(hf.e_tot),
        "e_ccsd": float(ccsd.e_tot),
        "e_fci": e_fci,
        "ccsd_t1": np.array(ccsd.t1),   # (n_ocupados, n_virtuales), orbitales espaciales
        "ccsd_t2": np.array(ccsd.t2),   # (o, o, v, v), orbitales espaciales
        "number_op": operador_numero(norb),
        "s2_op": operador_s2(norb),
    }


# --------------------------------------------------------------------------
# Validación
# --------------------------------------------------------------------------
def validate_problem(problem: dict, penalizacion: float = 10.0) -> dict:
    """Autovalor más chico de H restringido a N=4 y S²=0, comparado con el FCI de PySCF.

    Se suma una penalización a (N - 4)² y a S² para aislar el sector de 4 electrones singlete.
    Devuelve el error |E_min - e_fci| y los valores esperados de N y S² en ese estado.
    """
    H = problem["hamiltonian"].to_matrix()
    N = problem["number_op"].to_matrix()
    S2 = problem["s2_op"].to_matrix()
    ne = int(round(sum(int(b) for b in problem["hf_bitstring"])))
    I = np.eye(H.shape[0])
    pen = H + penalizacion * (N - ne * I) @ (N - ne * I) + penalizacion * S2
    w, v = np.linalg.eigh(pen)
    psi = v[:, 0]
    e_min = float(np.real(psi.conj() @ H @ psi))
    return {
        "e_min": e_min,
        "error": abs(e_min - problem["e_fci"]),
        "N": float(np.real(psi.conj() @ N @ psi)),
        "S2": float(np.real(psi.conj() @ S2 @ psi)),
        "e_hf_check": float(np.real(_estado_hf(problem).conj() @ H @ _estado_hf(problem))),
    }


def _estado_hf(problem):
    v = np.zeros(2 ** problem["n_qubits"], dtype=complex)
    v[int(problem["hf_bitstring"], 2)] = 1.0
    return v


if __name__ == "__main__":
    for d in (1.0, 1.8, 3.0):
        p = build_problem(d)
        r = validate_problem(p)
        print(f"d={d}: E_min={r['e_min']:.10f} FCI={p['e_fci']:.10f} "
              f"|dif|={r['error']:.2e} N={r['N']:.6f} S2={r['S2']:.2e} "
              f"E(HF qubits)={r['e_hf_check']:.8f} E(HF PySCF)={p['e_hf']:.8f}")
