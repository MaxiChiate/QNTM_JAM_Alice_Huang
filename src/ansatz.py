# Convención de qubits de Jordan-Wigner,
# qubits 0..norb-1 = spin-orbitales alfa, norb..2norb-1 = beta.
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_nature.second_q.circuit.library import PUCCD, UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper


def _occupied(problem):
    """Qubits ocupados en HF. Convención Qiskit (0 = último qubit)"""
    return [i for i, b in enumerate(problem["hf_bitstring"][::-1]) if b == "1"]


def _hf_circuit(problem):
    """Circuito de Hartree-Fock"""
    qc = QuantumCircuit(problem["n_qubits"])
    for i in _occupied(problem):
        qc.x(i)
    return qc


def build_ansatz(problem: dict, kind: str, k: int = 1) -> QuantumCircuit:
    """kind: 'hf' (solo HF), 'uccsd', 'uccgsd' (UCCSD generalizado) o 'kupccgsd'. Parten de HF.
    k = cuántas veces se repite el bloque (no aplica a 'hf')."""
    hf = _hf_circuit(problem)  # el circuito inicia en el estado de Hartree-Fock
    if kind == "hf":
        return hf

    # Orbitales espaciales y electrones alfa/beta leídos del estado HF
    norb = problem["n_qubits"] // 2
    n_alpha = sum(q < norb for q in _occupied(problem))
    n_beta = len(_occupied(problem)) - n_alpha
    args = (norb, (n_alpha, n_beta), JordanWignerMapper())

    if kind == "uccsd":
        ansatz = UCCSD(*args, reps=k, initial_state=hf)
    elif kind == "uccgsd":
        # Saltos entre cualquier par de orbitales (incluye ocupados a vacíos)
        ansatz = UCCSD(*args, reps=k, generalized=True, initial_state=hf)
    elif kind == "kupccgsd":
        # Dobles en pares (alfa y beta juntos) y simples, entre todos los orbitales, repetido k veces
        ansatz = PUCCD(*args, reps=k, include_singles=(True, True), generalized=True, initial_state=hf)
    else:
        raise ValueError(f"ansatz inválido: {kind!r}")

    # Se descompone en compuertas para simular rápido.
    return transpile(ansatz, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=0)


def count_cz(circuit: QuantumCircuit) -> int:
    """Transpila a ['cz','sx','x','rz'] con conectividad en línea y cuenta cz (costo de simulación)."""
    t = transpile(circuit, basis_gates=["cz", "sx", "x", "rz"],
                  coupling_map=CouplingMap.from_line(circuit.num_qubits),
                  optimization_level=1, seed_transpiler=0)
    return t.count_ops().get("cz", 0)
