# Poda de uccsd2_tap: en cada d se queda con los n saltos de mayor |θ| y re-optimiza
# (warm start desde esos θ). Guarda results/uccsd2_tap_p<n>_d<d>.json con el mismo formato
# que vqe.sweep, más "excitations" (los saltos que quedaron, cambian con d).
#   python tools/run_prune.py [n] [d ...]        (sin d: todas las distancias)
import json
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
ns = {}
for name in ["hamiltonian", "ansatz", "vqe", "measurement", "analysis", "pipeline"]:
    exec((root / "src" / f"{name}.py").read_text(encoding="utf-8"), ns)

SOURCE = "uccsd2_tap"
n_keep = int(sys.argv[1]) if len(sys.argv) > 1 else 20
distances = [float(d) for d in sys.argv[2:]] or ns["DISTANCES"]
name = f"{SOURCE}_p{n_keep}"

for d in distances:
    path = root / "results" / f"{name}_d{d}.json"
    if path.exists():
        print(f"d={d}: ya existe {path.name}", flush=True)
        continue
    row = json.load(open(root / "results" / f"{SOURCE}_d{d}.json"))
    problem, full = ns["prepare_point"](d, SOURCE)
    # θ con los signos de orbitales de esta corrida, para que el warm start sirva
    theta, _ = ns["align_theta"](problem, full, SOURCE, row["theta"], row["e_exact"])
    excitations = ns["excitation_list"](problem, "uccsd") * 2      # mismo orden que θ (2 repeticiones)
    keep = np.sort(np.argsort(-np.abs(theta))[:n_keep])
    kept = [excitations[i] for i in keep]
    ansatz = ns["build_ansatz"](problem, "custom", excitations=kept)
    r = ns["optimize"](problem, ansatz, theta0=theta[keep])
    out = {
        "d": d, "kind": name, "theta": r["theta"].tolist(), "e_exact": r["e_exact"],
        "n_iters": r["n_iters"], "cz": ns["count_cz"](ansatz), "n_qubits": problem["n_qubits"],
        "e_hf": problem["e_hf"], "e_ccsd": problem["e_ccsd"], "e_fci": problem["e_fci"],
        "excitations": [[list(occ), list(virt)] for occ, virt in kept],
    }
    json.dump(out, open(path, "w"), indent=1)
    print(f"d={d}: error {1000 * (out['e_exact'] - out['e_fci']):.4f} mHa, cz {out['cz']}, "
          f"evaluaciones {out['n_iters']}", flush=True)
