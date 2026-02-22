from __future__ import annotations
from typing import List


def accuracy_from_labels(pred: List[str], gold: List[str]) -> float:
    assert len(pred) == len(gold)
    correct = 0
    for p, g in zip(pred, gold):
        if p == g:
            correct += 1
    return correct / max(1, len(gold))