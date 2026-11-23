from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from ink_sdk.adapters.v8in import V8inAdapter
from ink_sdk.contracts import PreparedInput, RunContext, ScalePlan
from ink_sdk.profiles import ModelProfile
from ink_sdk.registry import get_profile


def test_relative_models_dir_still_finds_predict_py(tmp_path, monkeypatch):
    # predict.py runs with cwd = its own folder, so every path handed to it must not be relative to ours.
    monkeypatch.chdir(tmp_path)
    code = Path("models") / "ink-8um-v8in"
    (code / "ink8um").mkdir(parents=True)
    (code / "predict.py").write_text("")
    (code / "model.safetensors").write_bytes(b"fake")
    monkeypatch.setattr(ModelProfile, "verify_checkpoint", lambda self, models: Path(models) / "ink-8um-v8in" / "model.safetensors")
    calls = []

    def fake_run(command, cwd=None, **kwargs):
        calls.append(command)
        for flag in ("--model", "--layers"):
            assert (Path(cwd) / command[command.index(flag) + 1]).is_dir(), flag
        assert (Path(cwd) / command[1]).is_file()
        assert command[0] == "python3" or (Path(cwd) / command[0]).is_file()
        np.save(Path(cwd) / command[command.index("--output") + 1], np.zeros((8, 8), np.float32))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    prepared = PreparedInput(path=Path("p.zarr"), array=np.full((28, 8, 8), 100, dtype=np.uint8),
                             pixel_um=(9.362,) * 3, source=Path("p.zarr"),
                             scale_plan=ScalePlan(9.362, 9.362, "native"))
    Path("venv/bin").mkdir(parents=True)
    Path("venv/bin/python").write_text("")
    for python in ("venv/bin/python", "python3"):  # a relative path, and a bare name left to PATH
        calls.clear()
        ctx = RunContext(workdir=Path("work"), models_dir=Path("models"), device="cpu", python=python)
        raw = V8inAdapter().run(prepared, get_profile("v8in@ink-8um-v8in"), ctx)
        assert set(raw.maps) == {"forward", "reverse", "mean"} and len(calls) == 2
