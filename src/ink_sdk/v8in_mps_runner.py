"""Run v8-in's own predict.py with ink-sdk's MPS 3-D convolution path.

usage: python v8in_mps_runner.py PREDICT_PY [predict.py arguments ...]

predict.py stays the source of the tiling, stitching and layer choice. This only wraps ``ink8um.inference.predict_stack``
so that, when the device is MPS, the model's 3-D convolutions run through 2-D kernels (``ink_sdk.mps``) before the first
tile. On any other device it changes nothing. It is run as a file, not ``-m``, so the interpreter predict.py needs does
not have to have ink-sdk installed: ``mps.py`` is loaded from next to this file and needs only torch.
"""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path


def _load_mps():
    spec = importlib.util.spec_from_file_location("_ink_sdk_mps", Path(__file__).with_name("mps.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(__doc__)
    script = Path(argv[0]).resolve()
    sys.argv = [str(script), *argv[1:]]
    sys.path.insert(0, str(script.parent))
    import ink8um.inference as inference

    mps = _load_mps()
    original = inference.predict_stack

    def predict_stack(model, stack, *args, device=None, **kwargs):
        if device is not None and mps.speed_up_3d_convs_on_mps(model, device):
            print("ink-sdk: MPS 3-D convolutions through 2-D kernels", flush=True)
        return original(model, stack, *args, device=device, **kwargs)

    inference.predict_stack = predict_stack  # predict_surface looks it up in the module at call time
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
