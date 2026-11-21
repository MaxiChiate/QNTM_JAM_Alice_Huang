# Dueño: Maxi. Genera notebooks/final.ipynb (nbformat) a partir de:
# celda de instalación con versiones fijas + src/*.py en orden de dependencias
# (hamiltonian, ansatz, vqe, measurement, analysis) + notebooks/narrative/*.md.
# El .ipynb final se genera, no se edita a mano.
"""Build the self-contained Colab notebook.

Usage (from the repo root, with the project venv):
    .venv/bin/python tools/build_notebook.py            # writes notebooks/final.ipynb
    .venv/bin/python tools/build_notebook.py --check    # exit 1 if final.ipynb is out of date

Layout: every entry of SECTIONS is (name, source). For each one, the builder adds
    1. notebooks/narrative/<name>.md if it exists (markdown; split into several
       cells with a line containing only `<!-- cell -->`),
    2. the code of `source` if given and the file exists. A `# %%` line splits a
       .py file into several code cells. Top-level `if __name__ == "__main__":`
       blocks are dropped, because in a notebook __name__ is "__main__" and they
       would run.

Pinned versions are read from the environment that runs this script, so the jury
installs exactly what we developed with. numpy, scipy, matplotlib and pandas are
left to Colab's preinstalled versions: pinning them forces a runtime restart.
"""
import argparse
import ast
import glob
import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NARRATIVE_DIR = os.path.join(ROOT, "notebooks", "narrative")
OUTPUT = os.path.join(ROOT, "notebooks", "final.ipynb")

PINNED_PACKAGES = ["qiskit", "qiskit-aer", "qiskit-nature", "pyscf"]

TITLE = "# H₄ lineal con VQE: curva de energía, $R_e$ y $D_e$"

# Dependency order: hamiltonian -> ansatz -> vqe -> measurement -> analysis -> pipeline.
SECTIONS = [
    ("00_intro", None),
    ("10_hamiltonian", "src/hamiltonian.py"),
    ("20_ansatz", "src/ansatz.py"),
    ("30_vqe", "src/vqe.py"),
    ("40_measurement", "src/measurement.py"),
    ("50_analysis", "src/analysis.py"),
    ("60_pipeline", "src/pipeline.py"),
    ("70_results", "RUN"),                # generated: θ cache + measure the curve + report
    ("90_conclusions", None),
]

# Ansatz whose curve the notebook measures, and the shot budgets per point.
FINAL_KIND = "uccsd_tap"
TOTAL_SHOTS = 5_000_000
RELIABILITY_SHOTS = 1_000_000
RESULTS_DIR = os.path.join(ROOT, "results")

MARKDOWN_SPLIT = "<!-- cell -->"
CODE_SPLIT = "# %%"


def install_cell():
    pins = []
    for pkg in PINNED_PACKAGES:
        try:
            pins.append(f"{pkg}=={version(pkg)}")
        except PackageNotFoundError:
            sys.exit(f"{pkg} is not installed in this environment, cannot pin its version")
    return new_code_cell(
        "# Instalación de dependencias (versiones fijadas con las que desarrollamos)\n"
        f"%pip install -q {' '.join(pins)}"
    )


def timer_start_cell():
    return new_code_cell("import time\n_notebook_t0 = time.time()")


def timer_end_cell():
    return new_code_cell(
        'print(f"Tiempo total de ejecución: {(time.time() - _notebook_t0) / 60:.1f} min")'
    )


def theta_cache_cell(kind: str):
    """Fede's optimal θ (and their exact energies) embedded in the notebook: Colab has no results/."""
    points = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, f"{kind}_d*.json"))):
        with open(path) as f:
            row = json.load(f)
        if row.get("kind") == kind:
            points[float(row["d"])] = {"theta": row["theta"], "e_exact": row["e_exact"]}
    if not points:
        sys.exit(f"no results/{kind}_d*.json files to embed")
    lines = [f"# θ óptimos de {kind} (optimización con statevector, generado desde results/{kind}_d*.json)",
             f"FINAL_KIND = {kind!r}", "POINTS = {"]
    for d in sorted(points):
        theta = ", ".join(repr(round(t, 12)) for t in points[d]["theta"])
        lines.append(f"    {d!r}: {{\"e_exact\": {points[d]['e_exact']!r}, \"theta\": [{theta}]}},")
    lines.append("}")
    return new_code_cell("\n".join(lines))


def run_cells():
    return [
        theta_cache_cell(FINAL_KIND),
        new_code_cell(f"TOTAL_SHOTS = {TOTAL_SHOTS:_}\nRELIABILITY_SHOTS = {RELIABILITY_SHOTS:_}\n\n"
                      "records = measure_curve(FINAL_KIND, POINTS, TOTAL_SHOTS, reliability_shots=RELIABILITY_SHOTS)\n"
                      "out = report(records)"),
        new_code_cell('out["table"]'),
        new_code_cell('out["figure"]'),
        new_code_cell('out["summary"]'),
        new_code_cell('check = out["beats_ccsd"]\n'
                      'print("le gana a CCSD en todos los puntos con d ≥ 1,8 Å:", check["all_ok"])\n'
                      'print("precisión química en esa zona:", check["all_chemical_accuracy"])\n'
                      'pd.DataFrame(check["points"])'),
        new_code_cell("reliability_table(records)"),
    ]


def strip_main_blocks(code: str) -> str:
    """Remove top-level `if __name__ == "__main__":` blocks."""
    tree = ast.parse(code)
    lines = code.splitlines(keepends=True)
    drop = []
    for node in tree.body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"):
            drop.append((node.lineno - 1, node.end_lineno))
    for start, end in reversed(drop):
        del lines[start:end]
    return "".join(lines)


def split_cells(text: str, marker: str) -> list:
    chunks, current = [], []
    for line in text.splitlines():
        if line.strip() == marker:
            chunks.append("\n".join(current))
            current = []
        else:
            current.append(line)
    chunks.append("\n".join(current))
    return [c.strip("\n") for c in chunks if c.strip()]


def build() -> nbformat.NotebookNode:
    cells = [new_markdown_cell(TITLE), install_cell(), timer_start_cell()]
    for name, source in SECTIONS:
        md_path = os.path.join(NARRATIVE_DIR, f"{name}.md")
        if os.path.exists(md_path):
            with open(md_path, encoding="utf-8") as f:
                cells += [new_markdown_cell(c) for c in split_cells(f.read(), MARKDOWN_SPLIT)]
        if source is None:
            continue
        if source == "RUN":
            cells += run_cells()
            continue
        src_path = os.path.join(ROOT, source)
        if not os.path.exists(src_path):
            continue
        with open(src_path, encoding="utf-8") as f:
            code = strip_main_blocks(f.read())
        header = f"# ---- {source} ----\n"
        chunks = split_cells(code, CODE_SPLIT)
        if chunks:
            chunks[0] = header + chunks[0]
        cells += [new_code_cell(c) for c in chunks]
    cells.append(timer_end_cell())
    for i, cell in enumerate(cells):
        cell.id = f"cell-{i:03d}"   # deterministic ids, so rebuilding gives the same file

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
    return nb


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if the output differs from a fresh build")
    args = parser.parse_args()

    text = nbformat.writes(build())
    if args.check:
        current = open(args.output, encoding="utf-8").read() if os.path.exists(args.output) else ""
        if current != text:
            sys.exit(f"{args.output} is out of date, run tools/build_notebook.py")
        print(f"{args.output} is up to date")
        return
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
