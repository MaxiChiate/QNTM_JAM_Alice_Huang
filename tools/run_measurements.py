# Dueño: Maxi. Mide con shots la curva de un ansatz a partir de los θ de Fede.
#   python tools/run_measurements.py [kind] [shots] [reliability_shots] [stretched_shots]
#   kind: nombre de results/<kind>_d<d>.json (uccsd, uccsd_tap, uccsd2_tap, ...)
# Carga los src/*.py en un solo namespace, igual que el notebook de Colab, y guarda
# results/measured/<kind>_d<d>.json (los registros de analysis.make_record).
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
ns = {}
for name in ["hamiltonian", "ansatz", "vqe", "measurement", "analysis", "pipeline"]:
    exec((root / "src" / f"{name}.py").read_text(encoding="utf-8"), ns)

kind = sys.argv[1] if len(sys.argv) > 1 else "uccsd_tap"
shots = int(float(sys.argv[2])) if len(sys.argv) > 2 else 5_000_000
reliability_shots = int(float(sys.argv[3])) if len(sys.argv) > 3 else 1_000_000
stretched_shots = int(float(sys.argv[4])) if len(sys.argv) > 4 else None   # shots for d >= 1.8 Å

points = ns["load_points"](kind, str(root / "results"))
if not points:
    sys.exit(f"no results/{kind}_d*.json files")
t0 = time.time()
records = ns["measure_curve"](kind, points, shots, reliability_shots=reliability_shots,
                              stretched_shots=stretched_shots)
out_dir = str(root / "results" / "measured")
for rec in records:
    ns["save_record"](rec, out_dir)
print(f"{len(records)} points in {time.time() - t0:.0f} s -> {out_dir}")
