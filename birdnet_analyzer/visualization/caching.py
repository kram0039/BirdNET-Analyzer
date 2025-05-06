from __future__ import annotations
import hashlib, json, pathlib
from functools import lru_cache
from typing import Any, Tuple, FrozenSet

def _stable_hash(obj: Any) -> str:
    """
    JSON-stable hashing helper – turns *almost anything* that is JSON-serialisable
    (lists, dicts, strings, numbers) into an sha256 hex digest that is
    guaranteed to be identical for equal content, independent of ordering
    inside dicts.
    """
    dumped = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(dumped).hexdigest()

def fingerprint_prediction_dir(pred_dir: str, col_map: dict[str, str]) -> str:
    """
    → unique id for the **content** of the directory **and** the column
      mapping the user selected.
    """
    root = pathlib.Path(pred_dir)
    file_info: list[Tuple[str, float]] = [
        (p.name, p.stat().st_mtime)        # (file name, last modified)
        for p in sorted(root.glob("*.csv")) + sorted(root.glob("*.tsv"))
    ]
    return _stable_hash({"files": file_info, "columns": col_map})

def fingerprint_metadata_file(meta_file: str, col_map: dict[str, str]) -> str:
    """
    → unique id for the **content** (mtime) of the metadata file **and**
      its column mapping.
    """
    p = pathlib.Path(meta_file)
    return _stable_hash({
        "file": p.name,
        "mtime": p.stat().st_mtime,
        "columns": col_map
    })