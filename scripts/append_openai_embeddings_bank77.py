"""Add openai (Azure or OpenAI API) embeddings to processed_data/data_bank77.pkl if missing."""
from __future__ import annotations

import os
import pickle
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from kLLMmeans import get_embeddings  # noqa: E402


def main() -> int:
    path = os.path.join("processed_data", "data_bank77.pkl")
    if not os.path.isfile(path):
        print(f"Missing {path}", file=sys.stderr)
        return 1
    with open(path, "rb") as f:
        data_dict = pickle.load(f)
    emb = data_dict.get("embeddings") or {}
    if "openai" in emb:
        print("openai embeddings already present; skipping.")
        return 0
    docs = data_dict["documents"]
    print(f"Computing openai embeddings for {len(docs)} documents...")
    emb["openai"] = get_embeddings(docs, emb_type="openai")
    data_dict["embeddings"] = emb
    with open(path, "wb") as f:
        pickle.dump(data_dict, f)
    print(f"Updated {path} (openai shape {emb['openai'].shape})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
