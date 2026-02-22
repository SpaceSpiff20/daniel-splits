from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    results_dir = Path("results")
    files = sorted(results_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit("No results/*.jsonl found. Run an experiment first.")

    rows = []
    for fp in files:
        rows.extend(load_jsonl(fp))

    df = pd.DataFrame(rows)

    # Minimal sanity check
    required = {"model", "method", "dataset", "N"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    # Aggregate (mean across seeds if multiple)
    group_cols = ["model", "method", "dataset", "N"]
    agg_cols = {}
    for col in ["accuracy", "avg_latency_ms", "tokens_per_sec", "avg_prompt_tokens"]:
        if col in df.columns:
            agg_cols[col] = "mean"

    gdf = df.groupby(group_cols, as_index=False).agg(agg_cols)

    # --- Plot 1: Accuracy vs N ---
    if "accuracy" in gdf.columns:
        plt.figure()
        for (model, method, dataset), sub in gdf.groupby(["model", "method", "dataset"]):
            sub = sub.sort_values("N")
            plt.plot(sub["N"], sub["accuracy"], marker="o", label=f"{model} | {method} | {dataset}")
        plt.xlabel("N (shots / train examples)")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs N")
        plt.legend()
        plt.tight_layout()
        Path("analysis_out").mkdir(exist_ok=True)
        plt.savefig("analysis_out/accuracy_vs_N.png", dpi=200)
        plt.close()

    # --- Plot 2: Efficiency vs N (latency or tokens/sec) ---
    y_col = None
    y_label = None
    if "avg_latency_ms" in gdf.columns:
        y_col = "avg_latency_ms"
        y_label = "Avg Latency (ms)"
    elif "tokens_per_sec" in gdf.columns:
        y_col = "tokens_per_sec"
        y_label = "Tokens/sec"

    if y_col:
        plt.figure()
        for (model, method, dataset), sub in gdf.groupby(["model", "method", "dataset"]):
            sub = sub.sort_values("N")
            plt.plot(sub["N"], sub[y_col], marker="o", label=f"{model} | {method} | {dataset}")
        plt.xlabel("N (shots / train examples)")
        plt.ylabel(y_label)
        plt.title(f"{y_label} vs N")
        plt.legend()
        plt.tight_layout()
        Path("analysis_out").mkdir(exist_ok=True)
        plt.savefig(f"analysis_out/{y_col}_vs_N.png", dpi=200)
        plt.close()

    print("Saved plots to analysis_out/")

if __name__ == "__main__":
    main()