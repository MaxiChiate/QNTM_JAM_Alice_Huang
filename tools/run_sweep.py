# Corre el barrido VQE localmente: python tools/run_sweep.py [kind] [k] [tap]
#   kind: uccsd, uccgsd, kupccgsd   |   k: repeticiones del bloque   |   tap: usar el problema con menos qubits
# Carga los src/*.py en un solo namespace, igual que el notebook de Colab.
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
ns = {}
for name in ["hamiltonian", "ansatz", "vqe"]:
    exec((root / "src" / f"{name}.py").read_text(encoding="utf-8"), ns)
kind = sys.argv[1] if len(sys.argv) > 1 else "uccsd"
k = int(sys.argv[2]) if len(sys.argv) > 2 else 1
taper = len(sys.argv) > 3 and sys.argv[3] == "tap"
ns["sweep"](kind, ns["DISTANCES"], out_dir=str(root / "results"), k=k, taper=taper)
