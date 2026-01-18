from __future__ import annotations

import os
from typing import Any

from utils.report_builder import build_json


def write_json(path: str, obj: Any) -> None:
    """Write canonical JSON to `path` using `utils.report_builder.build_json`.

    `obj` is expected to be a mapping with keys `metadata` and `results`, or just `results`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # If caller passed a full pre-built dict, use it; otherwise try to adapt
    if isinstance(obj, dict) and "results" in obj and "metadata" in obj:
        json_str = build_json(obj["results"], obj["metadata"])
    elif isinstance(obj, dict) and "results" in obj:
        json_str = build_json(obj["results"], {})
    else:
        # assume obj itself is a results list
        json_str = build_json(obj, {})

    with open(path, "w", encoding="utf-8") as f:
        f.write(json_str)
