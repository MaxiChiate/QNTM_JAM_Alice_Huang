# Grafica el error contra FCI y las cz de cada método guardado en results/ (y en patches/*/results/).
# Uso: python tools/plot_errors.py [salida.png]
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
out = sys.argv[1] if len(sys.argv) > 1 else str(root / "results" / "errores_metodos.png")

# {método: {d: fila}}. Los de patches/ llevan "(parche)" en el nombre.
methods = defaultdict(dict)
for path in sorted(glob.glob(str(root / "results" / "*.json")) + glob.glob(str(root / "patches" / "*" / "results" / "*.json"))):
    row = json.load(open(path))
    label = row["kind"] + (" (parche)" if "patches" in path else "")
    methods[label][row["d"]] = row

fig, (ax, ax_cz) = plt.subplots(1, 2, figsize=(13, 5))
floor = 1e-4  # mHa; por debajo el error es ruido numérico y no se puede graficar en escala log
ccsd = {}
for label, rows in methods.items():
    ds = sorted(rows)
    err = [max(1000 * abs(rows[d]["e_exact"] - rows[d]["e_fci"]), floor) for d in ds]
    ax.plot(ds, err, "o-", label=label, ms=4)
    ax_cz.plot(ds, [rows[d]["cz"] for d in ds], "o-", label=label, ms=4)
    for d in ds:
        ccsd[d] = 1000 * abs(rows[d]["e_ccsd"] - rows[d]["e_fci"])

if ccsd:
    ds = sorted(ccsd)
    ax.plot(ds, [max(ccsd[d], floor) for d in ds], "k--", label="CCSD")
ax.axhline(1.6, color="gray", ls=":", label="precisión química (1,6 mHa)")
ax.axvspan(1.8, 3.0, color="orange", alpha=0.08, label="d ≥ 1,8 Å")
ax.set(yscale="log", xlabel="d (Å)", ylabel="|E − E_FCI| (mHa)", title=f"Error contra FCI (piso del gráfico: {floor:g} mHa)")
ax.legend(fontsize=8)
ax_cz.set(xlabel="d (Å)", ylabel="compuertas cz", title="Costo en cz")
ax_cz.legend(fontsize=8)
for a in (ax, ax_cz):
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out, dpi=130)
print("guardado:", out, "| métodos:", ", ".join(f"{m} ({len(r)} puntos)" for m, r in methods.items()))
