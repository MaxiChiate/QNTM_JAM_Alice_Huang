# Convención de qubits de Jordan-Wigner,
# qubits 0..norb-1 = spin-orbitales alfa, norb..2norb-1 = beta.
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Z2Symmetries
from qiskit.transpiler import CouplingMap
from qiskit_nature.second_q.circuit.library import PUCCD, UCC, UCCSD, HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper, TaperedQubitMapper


def _occupied(problem):
    """Qubits ocupados en HF. Convención Qiskit (0 = último qubit)"""
    return [i for i, b in enumerate(problem["hf_bitstring"][::-1]) if b == "1"]


def _hf_circuit(problem):
    """Circuito de Hartree-Fock"""
    qc = QuantumCircuit(problem["n_qubits"])
    for i in _occupied(problem):
        qc.x(i)
    return qc


def _orbitals(problem):
    """(orbitales espaciales, (e alfa, e beta), mapper). Con tapering vienen guardados en problem."""
    if "mapper" in problem:
        return problem["num_spatial_orbitals"], problem["num_particles"], problem["mapper"]
    norb = problem["n_qubits"] // 2
    n_alpha = sum(q < norb for q in _occupied(problem))
    return norb, (n_alpha, len(_occupied(problem)) - n_alpha), JordanWignerMapper()


def taper_problem(problem: dict) -> dict:
    """Problema con menos qubits usando las simetrías Z2 del hamiltoniano (H4: de 8 a 5 qubits).

    Se queda con el sector que tiene el estado HF."""
    H = problem["hamiltonian"]
    norb, particles, mapper = _orbitals(problem)
    z = Z2Symmetries.find_z2_symmetries(H)
    hf = problem["hf_bitstring"][::-1]
    # Las simetrías son cadenas de Z. El autovalor sobre HF es (-1)^(cantidad de Z sobre qubits ocupados)
    sector = [(-1) ** sum(c == "Z" and hf[q] == "1" for q, c in enumerate(s.to_label()[::-1])) for s in z.symmetries]
    z2 = Z2Symmetries(z.symmetries, z.sq_paulis, z.sq_list, sector)
    tapered_mapper = TaperedQubitMapper(mapper, z2)

    # Estado HF en los qubits que quedan
    hf_state = Statevector(HartreeFock(norb, particles, tapered_mapper))
    n = hf_state.num_qubits
    tapered = dict(problem, hamiltonian=z2.taper(H), n_qubits=n, mapper=tapered_mapper,
                   num_spatial_orbitals=norb, num_particles=particles,
                   hf_bitstring=format(int(np.argmax(abs(hf_state.data))), f"0{n}b"))
    for key in ("number_op", "s2_op"):  # N y S² conmutan con las simetrías
        if key in problem:
            tapered[key] = z2.taper(problem[key])
    return tapered


def _ucc(problem: dict, kind: str, k: int = 1, excitations=None):
    """Ansatz parte de HF."""
    hf = _hf_circuit(problem)  # el circuito inicia en el estado de Hartree-Fock
    norb, particles, mapper = _orbitals(problem)
    args = (norb, particles, mapper)

    if kind == "uccsd":
        return UCCSD(*args, reps=k, initial_state=hf)
    if kind == "uccgsd":
        # Saltos entre cualquier par de orbitales (incluye ocupados a vacíos)
        return UCCSD(*args, reps=k, generalized=True, initial_state=hf)
    if kind == "kupccgsd":
        # Dobles en pares (alfa y beta juntos) y simples, entre todos los orbitales, repetido k veces
        return PUCCD(*args, reps=k, include_singles=(True, True), generalized=True, initial_state=hf)
    if kind == "custom":
        # Solo los saltos de `excitations`, en ese orden (lo usa la poda)
        return UCC(norb, particles, excitations=lambda num_spatial_orbitals, num_particles: excitations, qubit_mapper=mapper, initial_state=hf)
    raise ValueError(f"ansatz inválido: {kind!r}")


def excitation_list(problem: dict, kind: str, k: int = 1) -> list:
    """Saltos (desde, hacia) de un ansatz, en el mismo orden que sus parámetros θ."""
    return _ucc(problem, kind, k).excitation_list


def build_ansatz(problem: dict, kind: str, k: int = 1, excitations=None) -> QuantumCircuit:
    """kind: 'hf' (solo HF), 'uccsd', 'uccgsd' (UCCSD generalizado), 'kupccgsd' o 'custom' (requiere `excitations`).
    k = cuántas veces se repite el bloque (no aplica a 'hf' ni a 'custom')."""
    if kind == "hf":
        return _hf_circuit(problem)
    ansatz = _ucc(problem, kind, k, excitations)

    # Se descompone en compuertas para simular rápido.
    return transpile(ansatz, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=0)


def count_cz(circuit: QuantumCircuit) -> int:
    """Transpila a ['cz','sx','x','rz'] con conectividad en línea y cuenta cz (costo de simulación)."""
    t = transpile(circuit, basis_gates=["cz", "sx", "x", "rz"],
                  coupling_map=CouplingMap.from_line(circuit.num_qubits),
                  optimization_level=1, seed_transpiler=0)
    return t.count_ops().get("cz", 0)
