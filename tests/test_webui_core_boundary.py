"""The Web UI imports only the core helpers listed in docs/architecture.md, and they stay light."""

from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBUI = ROOT / "webui"

ALLOWED_CORE_IMPORTS = {
    "data_input_layout",
    "platform_paths",
    "geneformer.gene_converter",
    "geneformer.ortholog_loss_gate",
}
CORE_TOP_LEVEL = {"geneformer", "platform_paths", "data_input_layout", "pipeline_lib", "state_feedback"} | {
    p.stem for p in (ROOT / "core").glob("*.py")
}
HEAVY = ("torch", "transformers", "datasets")


def _core_imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        found.update(n for n in names if n.split(".")[0] in CORE_TOP_LEVEL)
    return found


class TestWebuiCoreBoundary(unittest.TestCase):
    def test_webui_imports_only_allowed_core_modules(self):
        offending = {}
        for path in sorted(WEBUI.rglob("*.py")):
            extra = _core_imports(path) - ALLOWED_CORE_IMPORTS
            if extra:
                offending[str(path.relative_to(ROOT))] = sorted(extra)
        self.assertEqual(offending, {})

    def test_allowed_helpers_do_not_load_ml_stack(self):
        code = (
            "import sys\n"
            f"sys.path[:0] = [{str(ROOT / 'core')!r}, {str(ROOT / 'contracts')!r}, {str(WEBUI)!r}]\n"
            "import platform_paths, data_input_layout, ortholog_approval_ui, dropped_genes_ui\n"
            "import geneformer.gene_converter\n"
            f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        self.assertEqual(out, "")

    def test_lazy_package_exports_still_resolve(self):
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT / 'core')!r})\n"
            "import geneformer\n"
            "assert 'torch' not in sys.modules\n"
            "from geneformer import InSilicoPerturber, TranscriptomeTokenizer\n"
            "from geneformer import in_silico_perturber as isp\n"
            "assert isp.InSilicoPerturber is InSilicoPerturber\n"
            "assert geneformer.EmbExtractor.__module__ == 'geneformer.emb_extractor'\n"
            "try:\n"
            "    geneformer.no_such_name\n"
            "except AttributeError:\n"
            "    print('ok')\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        self.assertEqual(out, "ok")


if __name__ == "__main__":
    unittest.main()
