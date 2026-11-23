"""Studio (Priyank Aranke), run with the model repository's own inference code.

`_vendor/studio/model.py` and `inference.py` are byte-identical copies of the `hf/` folder of
priyankaranke/vesuvius-studio-multiscroll-cpt (MIT), so the tiling (128 px windows, central 96 px written) and the
input scaling (uint8 clipped at 200, divided by 255) are upstream's, not a second copy. This adapter only picks the
16 planes and the depth order, and calls `predict` once per direction.

Planes: the central 16 of the prepared stack, n//2 - 7 .. n//2 + 8 (upstream's recipe for a 31-layer render is
planes 23 down to 8: the same 16 planes, read deepest first). "forward" feeds them in stored order, "reverse"
deepest first, which is upstream's documented order for a vc_render_tifxyz render, hence `primary = "reverse"`.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np

from ..contracts import Issue, PreparedInput, RawOutput, RunContext, Segment
from ..errors import InkSDKError, ValidationError
from ..profiles import ModelProfile
from .base import BaseAdapter, batch_size, resolve_device, source_scale_issue, stats

PLANES = 16
VENDOR = Path(__file__).resolve().parents[1] / "_vendor" / "studio"


def _upstream():
    """Import the vendored inference module (it does `from model import ...`, so its folder goes on sys.path)."""
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    return importlib.import_module("inference")


def _default_model_factory(checkpoint: Path, device):
    return _upstream().load_model(checkpoint, device)


def _models_dir(ctx: RunContext | None) -> Path:
    if ctx is not None and ctx.models_dir:
        return Path(ctx.models_dir)
    return Path(os.environ.get("INK_SDK_MODELS", Path.home() / ".cache" / "ink-sdk" / "models"))


def planes(n_layers: int) -> list[int]:
    if n_layers < PLANES:
        raise InkSDKError(f"Studio needs {PLANES} layers, the prepared volume has {n_layers}")
    start = min(n_layers // 2 - 7, n_layers - PLANES)   # n//2 - 7 .. n//2 + 8; a 16-layer stack is used whole
    return list(range(start, start + PLANES))


class StudioAdapter(BaseAdapter):
    name = "studio"

    def __init__(self, model_factory: Callable | None = None, predict: Callable | None = None):
        self.model_factory = model_factory or _default_model_factory
        self.predict = predict

    def check(self, segment: Segment, profile: ModelProfile, ctx: RunContext | None = None) -> list[Issue]:
        issues: list[Issue] = []
        for module, hint in (("torch", "Studio needs torch"), ("safetensors", "Studio needs safetensors")):
            try:
                importlib.import_module(module)
            except ImportError:
                issues.append(Issue("error", f"missing-{module}", f"{hint}; pip install {module}"))
        checkpoint = profile.checkpoint_path(_models_dir(ctx))
        if not checkpoint.is_file():
            issues.append(Issue("error", "missing-checkpoint",
                                f"checkpoint not found: {checkpoint} (ink-sdk fetch --profile {profile.id})"))
        issue = source_scale_issue(segment, profile, ctx)
        if issue:
            issues.append(issue)
        return issues

    def run(self, prepared: PreparedInput, profile: ModelProfile, ctx: RunContext) -> RawOutput:
        torch = importlib.import_module("torch")
        volume = np.asarray(prepared.array)
        if volume.ndim != 3 or volume.dtype != np.uint8:
            raise ValidationError("Studio input must be uint8 ZYX")
        idx = planes(int(volume.shape[0]))
        stack = np.ascontiguousarray(volume[idx[0]:idx[-1] + 1])
        checkpoint = profile.verify_checkpoint(_models_dir(ctx))
        device = resolve_device(ctx.device, torch)
        amp = str(ctx.amp or profile.execution.amp).lower()
        amp = "bf16" if amp == "bf16" and device.startswith("cuda") else "fp32"
        stride = int(profile.input.stride or 64)
        blend = "none" if stride == 96 else str(profile.execution.blend)
        model = self.model_factory(checkpoint, device)
        from ..mps import prepare_for_device

        model = prepare_for_device(model, device)   # MPS: 3-D convolutions through 2-D kernels
        predict = self.predict or _upstream().predict
        direction = profile.execution.direction
        wanted = ["forward", "reverse"] if direction == "both" else [direction]
        started = time.time()
        maps: dict[str, np.ndarray] = {}
        with tempfile.TemporaryDirectory(prefix="ink-sdk-studio-", dir=ctx.workdir) as tmp:
            for d in wanted:
                vol = stack if d == "forward" else np.ascontiguousarray(stack[::-1])
                ctx.say(f"studio: {d}, planes {idx[0]}..{idx[-1]}, stride {stride}, {blend}, {amp}, {device}")
                out = predict(model, vol, Path(tmp) / f"{d}.npy", device, batch_size(profile, ctx), amp, stride, blend)
                maps[d] = np.array(out, dtype=np.float32)
                del out
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
        if len(maps) == 2:
            maps["mean"] = (maps["forward"] + maps["reverse"]) / 2.0
        return RawOutput(
            maps=maps,
            valid=stack.any(axis=0),
            quantisation={"source_dtype": "float32", "levels": None, "scale": 1.0},
            receipt={
                "adapter": "studio",
                "checkpoint": str(checkpoint),
                "planes": idx,
                "forward": "planes in stored order",
                "reverse": "planes deepest first (upstream's order for a vc_render_tifxyz render)",
                "stride": stride,
                "blend": blend,
                "batch": batch_size(profile, ctx),
                "amp": amp,
                "device": device,
                "direction": direction,
                "runtime_seconds": round(time.time() - started, 2),
                "statistics": {key: stats(value) for key, value in maps.items()},
                "non_claims": [
                    "A probability map is not OCR and does not prove ink or letters.",
                    "Studio weights are CC BY-NC 4.0 (non-commercial).",
                ],
            },
        )
