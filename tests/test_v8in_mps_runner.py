"""The v8in runner wraps the model repository's predict_stack: 2-D kernels on MPS, nothing on other devices."""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "src" / "ink_sdk" / "v8in_mps_runner.py"

# A stand-in for YoussefMoNader/ink-8um-v8in: predict.py calls ink8um.predict_surface, which looks up predict_stack.
INFERENCE = """
import torch
from torch import nn
def predict_stack(model, stack, *, device=None, **kwargs):
    return [type(m).__name__ for m in model.modules() if isinstance(m, nn.Conv3d)]
def predict_surface(model, layers_dir, **kwargs):
    return predict_stack(model, None, **kwargs)
"""
PREDICT = """
import sys
from torch import nn
from ink8um import predict_surface
device = sys.argv[sys.argv.index("--device") + 1]
model = nn.Sequential(nn.Conv3d(1, 2, 3, padding=1), nn.ReLU(), nn.Conv3d(2, 1, 1))
print("convs:", ",".join(predict_surface(model, "layers", device=device)))
"""


@pytest.fixture
def predict_py(tmp_path):
    package = tmp_path / "ink8um"
    package.mkdir()
    (package / "__init__.py").write_text("from .inference import predict_surface, predict_stack\n")
    (package / "inference.py").write_text(textwrap.dedent(INFERENCE))
    script = tmp_path / "predict.py"
    script.write_text(textwrap.dedent(PREDICT))
    return script


def _run(script, device, cwd):
    done = subprocess.run([sys.executable, str(RUNNER), str(script), "--device", device],
                          capture_output=True, text=True, cwd=cwd, timeout=120)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_runner_swaps_3d_convs_only_on_mps(predict_py, tmp_path):
    assert "convs: MPSConv3d,MPSConv3d" in _run(predict_py, "mps", tmp_path)
    assert "convs: Conv3d,Conv3d" in _run(predict_py, "cpu", tmp_path)
