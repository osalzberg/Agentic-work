from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class TestItem:
    id: str
    prompt: str
    expected_kql: str
    workspace_id: Optional[str]
    time_range: Optional[str]
    expected_output_path: Optional[str]
    expected_output_hash: Optional[str]
    required_columns: Optional[List[str]]
    required_operators: Optional[List[str]]
    tolerances: Optional[Dict[str, Any]]
    notes: Optional[str]


def _coerce_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"): return True
    if s in ("false", "0", "no"): return False
    return None


def load_dataset(path: str) -> List[TestItem]:
    items: List[TestItem] = []
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                items.append(TestItem(
                    id=obj.get("id", ""),
                    prompt=obj.get("prompt", ""),
                    expected_kql=obj.get("expected_kql", ""),
                    workspace_id=obj.get("workspace_id"),
                    time_range=obj.get("time_range"),
                    expected_output_path=obj.get("expected_output_path"),
                    expected_output_hash=obj.get("expected_output_hash"),
                    required_columns=obj.get("required_columns"),
                    required_operators=obj.get("required_operators"),
                    tolerances=obj.get("tolerances"),
                    notes=obj.get("notes")
                ))
    elif path.lower().endswith(".csv"):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for obj in reader:
                items.append(TestItem(
                    id=obj.get("id", ""),
                    prompt=obj.get("prompt", ""),
                    expected_kql=obj.get("expected_kql", ""),
                    workspace_id=obj.get("workspace_id"),
                    time_range=obj.get("time_range"),
                    expected_output_path=obj.get("expected_output_path"),
                    expected_output_hash=obj.get("expected_output_hash"),
                    required_columns=(obj.get("required_columns") or "").split("|") if obj.get("required_columns") else None,
                    required_operators=(obj.get("required_operators") or "").split("|") if obj.get("required_operators") else None,
                    tolerances=json.loads(obj.get("tolerances") or "{}"),
                    notes=obj.get("notes")
                ))
    else:
        raise ValueError(f"Unsupported dataset format: {path}")
    return items
