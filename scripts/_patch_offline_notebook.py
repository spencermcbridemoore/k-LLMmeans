"""Patch offline_experiments_modified.ipynb: all DATASETS + multi-dataset paper comparison."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "offline_experiments_modified.ipynb"

CELL0 = r'''from kLLMmeans import kLLMmeans, get_embeddings, summarize_cluster, spherical_kmeans_fit_predict
from experiment_utils import load_dataset, cluster_metrics, avg_closest_distance
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import numpy as np
import pandas as pd
import json
import pickle
import os

import warnings
warnings.filterwarnings("ignore")

# --- Config: offline baselines + paper comparison (Diaz-Rodriguez et al. 2025, text-embedding-3-small / openai) ---
# Preprocessed pickles: processed_data/data_<name>.pkl (see data_loaders/preprocess_offline_data.ipynb or scripts).
DATASETS = ["bank77", "clinic", "goemo", "massive_D", "massive_I"]
# Keys must exist in each pickle under ["embeddings"]. openai: Azure or OPENAI_KEY.
EMBEDDING_TYPES = ["distilbert", "e5-large", "sbert", "openai"]
RUN_KMEDOIDS = True
RUN_BROKEN_BASELINES = False  # GMM / spectral / agglomerative (uses StandardScaler + PCA)
RUN_LLM = False
NUM_SEEDS = 10
'''

CELL1 = r'''max_iter = 120

# baseline_results[data][emb_type][seed][method] -> dict with assignments, centroids, results (list)
baseline_results = {}

for data in DATASETS:
    pkl_path = os.path.join("processed_data", f"data_{data}.pkl")
    if not os.path.isfile(pkl_path):
        print(f"Skipping {data!r}: missing {pkl_path} — generate pickles first (e.g. preprocess notebook).")
        continue

    baseline_results[data] = {}
    with open(pkl_path, "rb") as f:
        data_dict = pickle.load(f)

    labels = data_dict["labels"]
    num_clusters = data_dict["num_clusters"]
    documents = data_dict["documents"]
    text_features = data_dict["embeddings"]
    prompt = data_dict["prompt"]
    text_type = data_dict["text_type"]
    oracle_cluster_assignments = labels

    for emb_type in EMBEDDING_TYPES:
        if emb_type not in text_features:
            print(f"Skipping {data}: missing embedding key {emb_type!r}")
            continue

        baseline_results[data][emb_type] = {}
        emb_data = text_features[emb_type]

        oracle_clustered_embeddings = {i: [] for i in range(num_clusters)}
        for embedding, cluster in zip(emb_data, oracle_cluster_assignments):
            oracle_clustered_embeddings[cluster].append(embedding)
        oracle_centroids = [
            np.mean(oracle_clustered_embeddings[i], axis=0) if oracle_clustered_embeddings[i] else None
            for i in range(num_clusters)
        ]
        oracle_summary_embeddings = oracle_centroids

        for seed in range(NUM_SEEDS):
            baseline_results[data][emb_type][seed] = {}

            # Euclidean k-means
            kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iter, random_state=seed)
            kmeans_assignments = kmeans.fit_predict(emb_data)
            kmeans_centroids = kmeans.cluster_centers_
            results = cluster_metrics(
                np.array(labels), kmeans_assignments, oracle_centroids, kmeans_centroids, oracle_summary_embeddings
            )
            baseline_results[data][emb_type][seed]["kmeans"] = {
                "assignments": kmeans_assignments,
                "final_centroids": kmeans_centroids,
                "results": results,
            }
            print([data, emb_type, seed, "kmeans", results])

            # Spherical k-means (not in paper as a reported baseline)
            kmeans_spherical_assignments, kmeans_spherical_centroids = spherical_kmeans_fit_predict(
                emb_data, num_clusters, max_iter=max_iter, random_state=seed, normalize=True
            )
            results = cluster_metrics(
                np.array(labels),
                kmeans_spherical_assignments,
                oracle_centroids,
                kmeans_spherical_centroids,
                oracle_summary_embeddings,
            )
            baseline_results[data][emb_type][seed]["kmeans_spherical"] = {
                "assignments": kmeans_spherical_assignments,
                "final_centroids": kmeans_spherical_centroids,
                "results": results,
            }
            print([data, emb_type, seed, "kmeans_spherical", results])

            if RUN_KMEDOIDS:
                kmedoids = KMedoids(n_clusters=num_clusters, max_iter=max_iter, random_state=seed)
                kmedoids.fit(emb_data)
                kmedoids_assignments = kmedoids.labels_
                kmedoids_indices = kmedoids.medoid_indices_
                kmedoids_centroids = emb_data[kmedoids_indices]
                results = cluster_metrics(
                    np.array(labels),
                    kmedoids_assignments,
                    oracle_centroids,
                    kmedoids_centroids,
                    oracle_summary_embeddings,
                )
                baseline_results[data][emb_type][seed]["kmedoids"] = {
                    "assignments": kmedoids_assignments,
                    "final_centroids": kmedoids_centroids,
                    "results": results,
                }
                print([data, emb_type, seed, "kmedoids", results])

            if RUN_BROKEN_BASELINES:
                X_scaled = StandardScaler().fit_transform(emb_data)

                pca = PCA(n_components=50, random_state=seed)
                X_pca = pca.fit_transform(X_scaled)
                gmm = GaussianMixture(n_components=num_clusters, random_state=seed)
                gmm_assignments = gmm.fit_predict(X_pca)
                gmm_clustered_embeddings = {i: [] for i in range(num_clusters)}
                for embedding, cluster in zip(emb_data, gmm_assignments):
                    gmm_clustered_embeddings[cluster].append(embedding)
                gmm_centroids = [
                    np.mean(gmm_clustered_embeddings[i], axis=0) if gmm_clustered_embeddings[i] else None
                    for i in range(num_clusters)
                ]
                results = cluster_metrics(
                    np.array(labels), gmm_assignments, gmm_centroids, oracle_centroids, oracle_centroids
                )
                baseline_results[data][emb_type][seed]["gmm"] = {
                    "assignments": gmm_assignments,
                    "results": results,
                }
                print([data, emb_type, seed, "gmm", results])

                spectral = SpectralClustering(
                    n_clusters=num_clusters, random_state=seed, affinity="nearest_neighbors"
                )
                spectral_assignments = spectral.fit_predict(X_scaled)
                spectral_clustered_embeddings = {i: [] for i in range(num_clusters)}
                for embedding, cluster in zip(emb_data, spectral_assignments):
                    spectral_clustered_embeddings[cluster].append(embedding)
                spectral_centroids = [
                    np.mean(spectral_clustered_embeddings[i], axis=0)
                    if spectral_clustered_embeddings[i]
                    else None
                    for i in range(num_clusters)
                ]
                results = cluster_metrics(
                    np.array(labels),
                    spectral_assignments,
                    spectral_centroids,
                    oracle_centroids,
                    oracle_centroids,
                )
                baseline_results[data][emb_type][seed]["spectral"] = {
                    "assignments": spectral_assignments,
                    "results": results,
                }
                print([data, emb_type, seed, "spectral", results])

                agglo = AgglomerativeClustering(n_clusters=num_clusters)
                agglo_assignments = agglo.fit_predict(X_scaled)
                agglo_clustered_embeddings = {i: [] for i in range(num_clusters)}
                for embedding, cluster in zip(emb_data, agglo_assignments):
                    agglo_clustered_embeddings[cluster].append(embedding)
                agglo_centroids = [
                    np.mean(agglo_clustered_embeddings[i], axis=0)
                    if agglo_clustered_embeddings[i]
                    else None
                    for i in range(num_clusters)
                ]
                results = cluster_metrics(
                    np.array(labels),
                    agglo_assignments,
                    agglo_centroids,
                    oracle_centroids,
                    oracle_centroids,
                )
                baseline_results[data][emb_type][seed]["agglomerative"] = {
                    "assignments": agglo_assignments,
                    "results": results,
                }
                print([data, emb_type, seed, "agglomerative", results])

            if RUN_LLM:
                for llm_type in [
                    "gpt-3.5-turbo",
                    "gpt-4o",
                    "llama3.3-70b",
                    "deepseek-chat",
                    "claude-3-7-sonnet-20250219",
                ]:
                    baseline_results[data][emb_type][seed][llm_type] = {}
                    for force_context_length in [0, 10]:
                        baseline_results[data][emb_type][seed][llm_type][force_context_length] = {}
                        for max_llm_iter in [1, 5]:
                            (
                                assignments,
                                final_summaries,
                                final_summary_embeddings,
                                final_centroids,
                                summaries_evolution,
                                centroids_evolution,
                            ) = kLLMmeans(
                                documents,
                                prompt=prompt,
                                text_type=text_type,
                                num_clusters=num_clusters,
                                force_context_length=force_context_length,
                                max_llm_iter=max_llm_iter,
                                max_iter=max_iter,
                                tol=1e-4,
                                random_state=seed,
                                emb_type=emb_type,
                                text_features=text_features[emb_type],
                            )
                            results = cluster_metrics(
                                np.array(labels),
                                assignments,
                                oracle_centroids,
                                final_centroids,
                                oracle_summary_embeddings,
                                final_summary_embeddings,
                            )
                            baseline_results[data][emb_type][seed][llm_type][force_context_length][
                                max_llm_iter
                            ] = {
                                "assignments": assignments,
                                "final_summaries": final_summaries,
                                "final_summary_embeddings": final_summary_embeddings,
                                "final_centroids": final_centroids,
                                "summaries_evolution": summaries_evolution,
                                "centroids_evolution": centroids_evolution,
                                "results": results,
                            }
                            print(
                                [
                                    data,
                                    emb_type,
                                    llm_type,
                                    seed,
                                    force_context_length,
                                    max_llm_iter,
                                    results,
                                ]
                            )

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"sims_offline_{data}_baseline.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(baseline_results[data], f)
    print(f"Saved {out_path}")
'''

CELL2 = r'''# Diaz-Rodriguez et al. 2025, arXiv:2502.09667v5, Table 5.
# text-embedding-3-small, 10 seeds — mean ACC/NMI; std in Table 5 parentheses.
# Paper targets below apply to emb_type "openai" only. Other embeddings: see Table 2/6 (different datasets);
# we do not duplicate those here — paper columns are NaN for non-openai rows.
# Spherical k-means is not reported in the paper.

def _paper_row(acc, acc_s, nmi, nmi_s):
    return {"acc_mean": acc, "acc_std": acc_s, "nmi_mean": nmi, "nmi_std": nmi_s}


# Keys match DATASETS / pickle names: bank77, clinic (CLINC), goemo, massive_D, massive_I
PAPER_TEXT_EMBEDDING_3_SMALL = {
    "bank77": {
        "kmeans": _paper_row(66.2, 1.82, 83.0, 0.56),
        "kmedoids": _paper_row(41.7, 0.0, 69.5, 0.0),
        "gmm": _paper_row(67.7, 1.78, 83.0, 0.54),
        "agglomerative": _paper_row(69.9, 0.0, 83.7, 0.0),
        "spectral": _paper_row(68.2, 0.56, 83.3, 0.38),
    },
    "clinic": {
        "kmeans": _paper_row(77.3, 1.83, 92.0, 0.35),
        "kmedoids": _paper_row(49.3, 0.0, 77.7, 0.0),
        "gmm": _paper_row(78.9, 1.18, 92.7, 0.22),
        "agglomerative": _paper_row(81.0, 0.0, 92.5, 0.0),
        "spectral": _paper_row(76.3, 0.401, 90.9, 0.14),
    },
    "goemo": {
        "kmeans": _paper_row(20.7, 1.0, 20.5, 0.67),
        "kmedoids": _paper_row(15.7, 0.0, 15.9, 0.0),
        "gmm": _paper_row(21.5, 0.82, 20.6, 0.47),
        "agglomerative": _paper_row(15.8, 0.0, 14.1, 0.0),
        "spectral": _paper_row(17.6, 0.32, 15.2, 0.32),
    },
    "massive_D": {
        "kmeans": _paper_row(59.4, 4.03, 67.9, 1.67),
        "kmedoids": _paper_row(37.6, 0.0, 38.8, 0.0),
        "gmm": _paper_row(56.2, 2.88, 68.6, 1.01),
        "agglomerative": _paper_row(62.8, 0.0, 67.1, 0.0),
        "spectral": _paper_row(61.5, 0.02, 67.0, 0.02),
    },
    "massive_I": {
        "kmeans": _paper_row(52.9, 1.07, 72.4, 0.54),
        "kmedoids": _paper_row(35.9, 0.0, 52.4, 0.0),
        "gmm": _paper_row(53.9, 1.91, 73.2, 0.80),
        "agglomerative": _paper_row(56.6, 0.0, 70.5, 0.0),
        "spectral": _paper_row(56.6, 0.84, 71.4, 0.23),
    },
}


def _load_all_baselines_if_needed():
    """Merge in-memory baseline_results (Cell 1) with per-dataset pickles under results/."""
    merged = {}
    br = globals().get("baseline_results")
    if not isinstance(br, dict):
        br = {}
    for d in DATASETS:
        if br.get(d):
            merged[d] = br[d]
            continue
        path = os.path.join("results", f"sims_offline_{d}_baseline.pkl")
        if os.path.isfile(path):
            with open(path, "rb") as f:
                merged[d] = pickle.load(f)
        else:
            print(
                f"WARN: No baseline for {d!r} (run Cell 1 after processed_data/data_{d}.pkl exists)."
            )
    if not merged:
        raise FileNotFoundError(
            "No baseline results loaded. Run Cell 1 first (with preprocessed pickles), or add results/*.pkl files."
        )
    return merged


def _load_bank77_baseline_if_needed():
    """Earlier notebooks used this name; it loads every `DATASETS` entry, not only Bank77."""
    return _load_all_baselines_if_needed()


def build_paper_comparison_table(all_results):
    """
    cluster_metrics returns ACC and NMI in [0, 1]; paper reports percentages (x100).
    """
    rows = []
    methods = ["kmeans", "kmeans_spherical"]
    if RUN_KMEDOIDS:
        methods.append("kmedoids")
    if RUN_BROKEN_BASELINES:
        methods.extend(["gmm", "spectral", "agglomerative"])

    for dataset in DATASETS:
        per_dataset = all_results.get(dataset) or {}
        if not per_dataset:
            continue
        for emb_type, seed_dict in sorted(per_dataset.items()):
            if not seed_dict:
                continue
            for method in methods:
                if not any(method in seed_dict.get(s, {}) for s in seed_dict):
                    continue
                accs = []
                nmis = []
                for seed in range(NUM_SEEDS):
                    if seed not in seed_dict or method not in seed_dict[seed]:
                        continue
                    r = seed_dict[seed][method]["results"]
                    accs.append(float(r[0]) * 100.0)
                    nmis.append(float(r[1]) * 100.0)
                if not accs:
                    continue

                acc_mean = float(np.mean(accs))
                acc_std = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
                nmi_mean = float(np.mean(nmis))
                nmi_std = float(np.std(nmis, ddof=1)) if len(nmis) > 1 else 0.0

                paper = None
                if emb_type == "openai":
                    paper = PAPER_TEXT_EMBEDDING_3_SMALL.get(dataset, {}).get(method)
                if paper is None:
                    paper_acc_m = paper_nmi_m = np.nan
                    paper_acc_s = paper_nmi_s = np.nan
                    delta_acc = np.nan
                else:
                    paper_acc_m = paper["acc_mean"]
                    paper_acc_s = paper["acc_std"]
                    paper_nmi_m = paper["nmi_mean"]
                    paper_nmi_s = paper["nmi_std"]
                    delta_acc = acc_mean - paper_acc_m

                rows.append(
                    {
                        "dataset": dataset,
                        "emb_type": emb_type,
                        "method": method,
                        "acc_mean": acc_mean,
                        "acc_std": acc_std,
                        "nmi_mean": nmi_mean,
                        "nmi_std": nmi_std,
                        "paper_acc_mean": paper_acc_m,
                        "paper_acc_std": paper_acc_s,
                        "paper_nmi_mean": paper_nmi_m,
                        "paper_nmi_std": paper_nmi_s,
                        "delta_acc_vs_paper": delta_acc,
                    }
                )
    return pd.DataFrame(rows)


all_runs = _load_bank77_baseline_if_needed()
comparison_df = build_paper_comparison_table(all_runs)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)

from IPython.display import display

print(
    f"Baselines loaded: {sorted(all_runs)} ({len(all_runs)} datasets) | {len(comparison_df)} metric rows.",
    flush=True,
)
for d in DATASETS:
    sub = comparison_df[comparison_df["dataset"] == d]
    if sub.empty:
        print(
            f"\\n[{d}] (no rows -- need results/sims_offline_{d}_baseline.pkl or re-run the experiment cell)",
            flush=True,
        )
        continue
    print(f"\\n### {d}", flush=True)
    display(sub.sort_values(["emb_type", "method"], ignore_index=True))

comparison_df.sort_values(["dataset", "emb_type", "method"], ignore_index=True)
'''


def to_source_lines(s: str) -> list:
    lines = s.splitlines(keepends=True)
    if not lines:
        return []
    return lines


def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    nb["cells"][0]["source"] = to_source_lines(CELL0)
    nb["cells"][1]["source"] = to_source_lines(CELL1)
    nb["cells"][2]["source"] = to_source_lines(CELL2)
    # Clear stale outputs for cleaner diff
    for i in range(3):
        nb["cells"][i]["outputs"] = []
        nb["cells"][i]["execution_count"] = None
    NB.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Updated", NB)


if __name__ == "__main__":
    main()
