"""
Build processed_data/data_<dataset>.pkl for all static benchmarks (offline_experiments_modified DATASETS).

Requires: experiment_utils data files (goemo.tsv, data_loaders/massive.jsonl), HF datasets,
and for openai embeddings either Azure or OPENAI_KEY (see kLLMmeans.py).

Usage (from repo root, with k-llmmeans env):
  python scripts/build_processed_pickles.py
  python scripts/build_processed_pickles.py --only clinic goemo
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from experiment_utils import load_dataset  # noqa: E402
from kLLMmeans import get_embeddings  # noqa: E402

DEFAULT_DATASETS = ["bank77", "clinic", "goemo", "massive_D", "massive_I"]
EMB_TYPES = ["distilbert", "openai", "e5-large", "sbert"]


def _load_one(name: str):
    if name.startswith("massive_"):
        return load_dataset(name, opt=name[-1])
    return load_dataset(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Subset of dataset names (default: all five benchmarks)",
    )
    args = ap.parse_args()
    names = args.only if args.only else DEFAULT_DATASETS
    os.makedirs("processed_data", exist_ok=True)
    for data in names:
        print(data)
        labels, documents, num_clusters, prompt, text_type, instructor_prompt = _load_one(data)
        embeddings = {}
        for emb_type in EMB_TYPES:
            print(" ", emb_type)
            try:
                embeddings[emb_type] = get_embeddings(
                    list(documents), emb_type=emb_type, instructor_prompt=""
                )
            except Exception as e:
                print(f"  FAIL {emb_type}: {e}", file=sys.stderr)
                raise
        out = {
            "data": data,
            "labels": labels,
            "num_clusters": num_clusters,
            "documents": list(documents),
            "embeddings": embeddings,
            "prompt": prompt,
            "text_type": text_type,
        }
        path = os.path.join("processed_data", f"data_{data}.pkl")
        with open(path, "wb") as f:
            pickle.dump(out, f)
        print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
