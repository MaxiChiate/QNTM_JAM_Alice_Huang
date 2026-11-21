# Dueño: Gonza. Solo librerías externas, nunca `from src.x import ...` (ver docs/task_split.md).

DISTANCES = []  # grilla de d en Å: 12-15 puntos en [0.6, 3.0], incluyendo 1.8 y 3.0


def build_problem(d: float) -> dict:
    """H4 lineal, STO-3G, separación d (Å).

    Devuelve: d, hamiltonian (SparsePauliOp, con repulsión nuclear), n_qubits,
    hf_bitstring (convención Qiskit), e_hf, e_ccsd, e_fci, ccsd_t1, ccsd_t2,
    number_op (N=4), s2_op (S²=0).
    """
    return NotImplemented
