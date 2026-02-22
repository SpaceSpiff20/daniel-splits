# src/adapters/superglue_boolq.py

from __future__ import annotations
from typing import Any, Dict, List
from datasets import load_dataset


def load_examples(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Standardize SuperGLUE BoolQ into the repo's unified schema:
      { "id", "x", "y", "label", "meta" }
    """
    ds = load_dataset(
        cfg["hf_id"],
        cfg["hf_config"],
        split=cfg["hf_split"],
        trust_remote_code=cfg.get("trust_remote_code", False),
    )

    # BoolQ uses:
    # - passage: str
    # - question: str
    # - answer: bool
    label_map = cfg.get("label_map", {"0": "False", "1": "True"})

    examples: List[Dict[str, Any]] = []
    for i, row in enumerate(ds):
        passage = row["passage"]
        question = row["question"]
        answer_bool = bool(row["answer"])

        # map bool -> "0"/"1" -> verbalizer
        label_id = "1" if answer_bool else "0"
        y = label_map[label_id]

        x = f"Passage: {passage}\nQuestion: {question}"

        ex = {
            "id": f"superglue_boolq_{cfg['hf_split']}_{i:06d}",
            "x": x,
            "y": y,
            "label": y,  # important for stratified sampling (label_key = "label")
            "meta": {
                "dataset": "superglue_boolq",
                "split": cfg["hf_split"],
                "row_idx": i,
                "label_id": int(label_id),
            },
        }
        examples.append(ex)

    return examples