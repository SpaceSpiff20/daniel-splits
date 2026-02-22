from __future__ import annotations
from pathlib import Path
from typing import Union
from .results_schema import RunResult


def append_jsonl(out_path: Union[str, Path], result: RunResult) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if result.timestamp == 0.0:
        result.timestamp = RunResult.now_ts()
    with out_path.open("a", encoding="utf-8") as f:
        f.write(result.to_json() + "\n")