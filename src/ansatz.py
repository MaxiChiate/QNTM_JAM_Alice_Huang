# Convención de qubits de Jordan-Wigner,
# qubits 0..norb-1 = spin-orbitales alfa, norb..2norb-1 = beta.
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_nature.second_q.circuit.library import UCCSD
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


def build_ansatz(problem: dict, kind: str) -> QuantumCircuit:
    """kind: 'hf' (solo HF, sin parámetros) o 'uccsd' (UCCSD, parte de HF)."""
    hf = _hf_circuit(problem)  # el circuito inicia en el estado de Hartree-Fock
    if kind == "hf":
        return hf
    if kind != "uccsd":
        raise ValueError(f"ansatz inválido: {kind!r}")

    # Orbitales espaciales y electrones alfa/beta, leídos del estado HF
    norb = problem["n_qubits"] // 2
    n_alpha = sum(q < norb for q in _occupied(problem))
    n_beta = len(_occupied(problem)) - n_alpha
    uccsd = UCCSD(norb, (n_alpha, n_beta), JordanWignerMapper(), initial_state=hf)

    # Se descompone en compuertas básicas una sola vez para que simular sea rápido.
    return transpile(uccsd, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=0)


def count_cz(circuit: QuantumCircuit) -> int:
    """Transpila a ['cz','sx','x','rz'] con conectividad en línea y cuenta cz (costo de simulación)."""
    t = transpile(circuit, basis_gates=["cz", "sx", "x", "rz"],
                  coupling_map=CouplingMap.from_line(circuit.num_qubits),
                  optimization_level=1, seed_transpiler=0)
    return t.count_ops().get("cz", 0)
