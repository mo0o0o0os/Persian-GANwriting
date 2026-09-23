"""Execute main.ipynb offline with tiny models and fake audio (QUICK_TEST), and report errors.

    python tests/make_test_assets.py <author_repo> <assets>
    python tests/run_notebook_smoke.py <assets> <workdir>
"""

import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parents[1]
assets, workdir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
workdir.mkdir(parents=True, exist_ok=True)
nb = nbformat.read(HERE / "main.ipynb", as_version=4)
settings = next(c for c in nb.cells if c.cell_type == "code" and "QUICK_TEST = False" in c.source)
overrides = {k: str(assets / "models" / k) for k in ("facebook/wav2vec2-base", "bert-base-uncased", "HooshvareLab/bert-fa-base-uncased")}
for old, new in {"QUICK_TEST = False": "QUICK_TEST = True",
                 'CODE_SOURCE = "github"': f"CODE_SOURCE = {str(HERE)!r}",
                 "EXTRA_DATA_DIRS = []": f"EXTRA_DATA_DIRS = [{str(assets / 'shemo_fake')!r}]",
                 "MODEL_OVERRIDES = {}": f"MODEL_OVERRIDES = {overrides!r}",
                 "RUN_EXTENSION_EXAMPLE = False": "RUN_EXTENSION_EXAMPLE = True"}.items():
    assert old in settings.source or old.startswith("RUN_EXT"), old
    settings.source = settings.source.replace(old, new)
for c in nb.cells:
    if c.cell_type == "code" and "RUN_EXTENSION_EXAMPLE = False" in c.source:
        c.source = c.source.replace("RUN_EXTENSION_EXAMPLE = False", "RUN_EXTENSION_EXAMPLE = True")

start = time.time()
client = NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(workdir)}})
try:
    client.execute()
    print(f"NOTEBOOK OK in {time.time() - start:.0f}s")
    ok = True
except Exception as err:
    print(f"NOTEBOOK FAILED after {time.time() - start:.0f}s: {type(err).__name__}: {str(err)[-3000:]}")
    ok = False
finally:
    nbformat.write(nb, workdir / "executed_main.ipynb")
for i, c in enumerate(nb.cells):
    for o in c.get("outputs", []):
        if o.output_type == "stream":
            for line in o.text.splitlines():
                if re.match(r"^\[\d\d:\d\d:\d\d\]", line):
                    print(f"cell {i}: {line[:220]}")
        elif o.output_type == "error":
            print(f"cell {i} ERROR {o.ename}: {o.evalue[:800]}")
sys.exit(0 if ok else 1)
