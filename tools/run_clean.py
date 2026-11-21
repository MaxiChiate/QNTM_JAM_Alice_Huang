# Dueño: Maxi. Ensayo local de "Reiniciar y ejecutar todo" en Colab (ítem 7 de docs/task_split.md).
"""Run notebooks/final.ipynb end to end in a clean environment, the way Colab would.

    python3 tools/run_clean.py            # reuse .colab_env/ if it exists
    python3 tools/run_clean.py --fresh    # recreate it from scratch

What it does:
  1. Creates a virtualenv in .colab_env/ with only what Colab ships preinstalled
     (numpy, scipy, matplotlib, pandas) plus the Jupyter machinery. No qiskit, no
     pyscf: the notebook's own `%pip install` cell has to bring them, as in Colab.
  2. Copies the notebook to an empty directory, so it cannot see src/ or results/.
     If it secretly depends on the repo, it fails here.
  3. Executes every cell and saves the executed copy as
     .colab_env/final_executed.ipynb, then prints the wall time of the whole run
     and the time of the slowest cells.

This is a proxy, not the real thing: Colab's package versions, CPU and Python
differ. The real check is still uploading the notebook to Colab and running
"Entorno de ejecución > Reiniciar y ejecutar todo".

Only uses the standard library, so it runs with any python3 (no project venv needed).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import venv
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_DIR = os.path.join(ROOT, ".colab_env")
NOTEBOOK = os.path.join(ROOT, "notebooks", "final.ipynb")
EXECUTED = os.path.join(ENV_DIR, "final_executed.ipynb")
COLAB_BASE = ["numpy", "scipy", "matplotlib", "pandas", "ipykernel", "nbclient", "nbformat"]
CELL_TIMEOUT = 3600  # seconds per cell


def env_python():
    return os.path.join(ENV_DIR, "bin", "python")


def make_env(fresh: bool):
    if fresh and os.path.exists(ENV_DIR):
        shutil.rmtree(ENV_DIR)
    if os.path.exists(env_python()):
        return
    print(f"creating clean environment in {ENV_DIR} ...")
    venv.create(ENV_DIR, with_pip=True)
    subprocess.run([env_python(), "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
    subprocess.run([env_python(), "-m", "pip", "install", "-q", *COLAB_BASE], check=True)


# Runs inside the clean env: executes the notebook and records the time of every cell.
RUNNER = r"""
import json, sys, time
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

src, dst, timeout = sys.argv[1], sys.argv[2], int(sys.argv[3])
nb = nbformat.read(src, as_version=4)
client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": "."}})
times, error = [], None
with client.setup_kernel():
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        t0 = time.time()
        try:
            client.execute_cell(cell, i)
        except CellExecutionError as e:
            error = {"cell": i, "message": str(e)[-3000:]}
            times.append((i, time.time() - t0, cell.source.splitlines()[0][:70]))
            break
        times.append((i, time.time() - t0, cell.source.splitlines()[0][:70]))
nbformat.write(nb, dst)
print(json.dumps({"times": times, "error": error}))
"""


def run_notebook():
    workdir = tempfile.mkdtemp(prefix="colab_run_")
    nb_copy = os.path.join(workdir, "final.ipynb")
    shutil.copy(NOTEBOOK, nb_copy)
    runner = os.path.join(ENV_DIR, "_runner.py")
    with open(runner, "w") as f:
        f.write(RUNNER)
    t0 = time.time()
    proc = subprocess.run([env_python(), runner, nb_copy, EXECUTED, str(CELL_TIMEOUT)],
                          cwd=workdir, capture_output=True, text=True)
    total = time.time() - t0
    shutil.rmtree(workdir, ignore_errors=True)
    if proc.returncode != 0:
        print(proc.stderr[-3000:])
        sys.exit("the runner itself failed")
    return json.loads(proc.stdout.strip().splitlines()[-1]), total


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fresh", action="store_true", help="recreate the clean environment")
    args = parser.parse_args()
    if not os.path.exists(NOTEBOOK):
        sys.exit("notebooks/final.ipynb does not exist, run tools/build_notebook.py first")

    make_env(args.fresh)
    result, total = run_notebook()

    print(f"\n{datetime.now():%Y-%m-%d %H:%M}  total: {total / 60:.1f} min "
          f"(includes the %pip install cell)")
    print("slowest cells:")
    for i, t, first_line in sorted(result["times"], key=lambda x: -x[1])[:5]:
        print(f"  cell {i:3d}  {t:7.1f} s  {first_line}")
    print(f"executed notebook: {EXECUTED}")
    if result["error"]:
        print(f"\nFAILED at cell {result['error']['cell']}:\n{result['error']['message']}")
        sys.exit(1)
    print("OK: every cell ran")


if __name__ == "__main__":
    main()
