# Convención de qubits de Jordan-Wigner,
# qubits 0..norb-1 = spin-orbitales alfa, norb..2norb-1 = beta.
from itertools import combinations, product
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap


def _occupied(problem):
    """Qubits ocupados en HF. Convención Qiskit (0 = último qubit)"""
    return [i for i, b in enumerate(problem["hf_bitstring"][::-1]) if b == "1"]


def _destruction(p, n):
    """Operador de destrucción del spin-orbital p con Jordan-Wigner"""
    return SparsePauliOp.from_sparse_list(
        [("Z" * p + "X", list(range(p + 1)), 0.5), 
         ("Z" * p + "Y", list(range(p + 1)), 0.5j)], 
        n)


def _jumps(norb, occ):
    """Saltos de uno y dos electrones desde HF, conservando N y Sz.
    Salto es (desde, hacia). Los qubits de donde salen los electrones y a donde llegan.
    """
    # Qubits ocupados y vacíos en HF, separados por spin
    occ_alpha = [q for q in occ if q < norb]
    occ_beta = [q for q in occ if q >= norb]
    vac_alpha = [q for q in range(norb) if q not in occ_alpha]
    vac_beta = [q for q in range(norb, 2 * norb) if q not in occ_beta]
    por_spin = [(occ_alpha, vac_alpha), (occ_beta, vac_beta)]

    jumps = []

    # Simples: un electrón pasa de un ocupado i a un vacío a, del mismo spin
    for occ_spin, vac_spin in por_spin:
        for i, a in product(occ_spin, vac_spin):
            jumps.append(((i,), (a,)))

    # Dobles con spin opuesto: Un electrón alfa y uno beta saltan a la vez
    for i, j, a, b in product(occ_alpha, occ_beta, vac_alpha, vac_beta):
        jumps.append(((i, j), (a, b)))

    #Dobles con el mismo spin: dos electrones del mismo spin saltan a la vez
    for occ_spin, vac_spin in por_spin:
        for (i, j), (a, b) in product(combinations(occ_spin, 2), combinations(vac_spin, 2)):
            jumps.append(((i, j), (a, b)))

    return jumps


def _hf_circuit(problem):
    #Circuito de Hartree-Fock
    qc = QuantumCircuit(problem["n_qubits"])
    for i in _occupied(problem):
        qc.x(i)
    return qc


def _jump_generator(src, dst, destroy, n):
    """Operador G de un salto, hermítico, para la compuerta exp(-i·θ·G).

    T mueve electrones de los qubits src a los qubits dst: primero destruye en src y después crea en dst.
    G = i(T - T†), así que exp(-i·θ·G) = exp(θ(T - T†)) es unitaria, y con θ=0 no hace nada.
    """
    T = SparsePauliOp("I" * n)
    for p in dst:
        T = T.dot(destroy[p].adjoint())  # a† crea un electrón en p
    for p in reversed(src):
        T = T.dot(destroy[p])            # a destruye un electrón en p
    return (1j * (T - T.adjoint())).simplify()


def build_ansatz(problem: dict, kind: str) -> QuantumCircuit:
    """kind: 'hf' (solo el estado HF, sin parámetros) o 'uccsd' (UCCSD con JW, parte de HF)."""
    n = problem["n_qubits"]
    qc = _hf_circuit(problem)  # el circuito arranca en el estado de Hartree-Fock
    if kind == "hf":
        return qc
    if kind != "uccsd":
        raise ValueError(f"ansatz inválido: {kind!r}")

    # UCCSD: una compuerta por salto, cada una con su propio ángulo θ
    destroy = [_destruction(p, n) for p in range(n)]
    jumps = _jumps(n // 2, _occupied(problem))
    thetas = ParameterVector("θ", len(jumps))
    for theta, (src, dst) in zip(thetas, jumps):
        G = _jump_generator(src, dst, destroy, n)
        qc.append(PauliEvolutionGate(G, time=theta), range(n))

    # Se descompone en compuertas básicas una sola vez para que simular sea rápido.
    return transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=0)


def count_cz(circuit: QuantumCircuit) -> int:
    """Transpila a ['cz','sx','x','rz'] con conectividad en línea y cuenta cz (costo de simulación)."""
    t = transpile(circuit, basis_gates=["cz", "sx", "x", "rz"],
                  coupling_map=CouplingMap.from_line(circuit.num_qubits),
                  optimization_level=1, seed_transpiler=0)
    return t.count_ops().get("cz", 0)
