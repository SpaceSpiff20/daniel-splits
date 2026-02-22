from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import time
import json


@dataclass
class RunResult:
    # Identity
    model: str              # e.g., "qwen3-4b"
    method: str             # e.g., "icl", "lora", "ptuning", "reft"
    dataset: str            # e.g., "boolq", "rte", "finpb"
    task: str               # e.g., "binary_cls"
    N: int                  # number of few-shot examples used in prompt or train set
    seed: int = 0

    # Metrics (fill what you have; leave others None)
    accuracy: Optional[float] = None
    exact_match: Optional[float] = None

    avg_latency_ms: Optional[float] = None         # end-to-end avg per example
    tokens_per_sec: Optional[float] = None         # generation throughput if measured
    avg_prompt_tokens: Optional[int] = None
    avg_output_tokens: Optional[int] = None

    # Bookkeeping
    num_eval_examples: Optional[int] = None
    timestamp: float = 0.0
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Remove empty keys to keep JSONL clean
        return {k: v for k, v in d.items() if v is not None}

    @staticmethod
    def now_ts() -> float:
        return time.time()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)