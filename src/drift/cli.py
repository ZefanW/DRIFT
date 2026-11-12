"""Command-line entry points for the complete DRIFT selection pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .io import fresh_directory, read_jsonl, sha256_file, write_json, write_jsonl


def run_score(corpus_features, anchor_features, output, fit_mode="stable", clip=None, zscore=False):
    from .features import load_feature_table
    from .scoring import task_centroids, score_features, debias_scores
    training, records, train_manifest = load_feature_table(corpus_features)
    anchors, metadata, anchor_manifest = load_feature_table(anchor_features)
    if train_manifest["feature_fingerprint"] != anchor_manifest["feature_fingerprint"]:
        raise ValueError("Corpus and anchors must use exactly the same checkpoint, mask and projection")
    centroid = task_centroids(anchors, metadata)
    out = fresh_directory(output)
    raw = np.lib.format.open_memmap(out / "raw.npy", mode="w+", dtype=np.float64,
                                   shape=(len(training), len(centroid["tasks"])))
    norms = np.lib.format.open_memmap(out / "norms.npy", mode="w+", dtype=np.float64, shape=(len(training),))
    for start in range(0, len(training), 1024):
        result = score_features(training[start:start+1024], centroid["centroids"])
        raw[start:start+len(result["raw"])] = result["raw"]
        norms[start:start+len(result["norms"])] = result["norms"]
    raw.flush(); norms.flush()
    corrected = debias_scores(raw, norms, fit_mode, clip_percentiles=clip, zscore=zscore)
    np.save(out / "centroids.npy", centroid["centroids"], allow_pickle=False)
    np.save(out / "corrected.npy", corrected["corrected"], allow_pickle=False)
    np.save(out / "scores.npy", corrected["scores"], allow_pickle=False)
    write_jsonl(out / "records.jsonl", records)
    report = {"schema_version": 1, "state": "complete", "rows": len(training),
              "tasks": centroid["tasks"], "centroid_diagnostics": centroid["diagnostics"],
              "fit_mode": fit_mode, "betas": corrected["betas"].tolist(), "fits": corrected["fits"],
              "diagnostics": corrected["diagnostics"], "input_sha256": train_manifest["input_sha256"],
              "feature_fingerprint": train_manifest["feature_fingerprint"],
              "scores_sha256": sha256_file(out / "scores.npy"),
              "records_sha256": sha256_file(out / "records.jsonl"),
              "excluded_corpus_rows": train_manifest["rejected_rows"]}
    write_json(out / "manifest.json", report)
    return report


def run_select(scores_dir, output, fraction=.1, corpus=None):
    from .data import export_subset
    from .scoring import select_top_fraction
    source = Path(scores_dir)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("state") != "complete":
        raise ValueError("Incomplete score table")
    for filename, key in [("scores.npy", "scores_sha256"), ("records.jsonl", "records_sha256")]:
        if sha256_file(source / filename) != manifest[key]:
            raise ValueError(f"Score artifact checksum mismatch: {filename}")
    scores = np.load(source / "scores.npy", allow_pickle=False)
    records = list(read_jsonl(source / "records.jsonl"))
    if len(scores) != len(records) or len(scores) != manifest["rows"]:
        raise ValueError("Score/record count mismatch")
    if corpus and sha256_file(corpus) != manifest["input_sha256"]:
        raise ValueError("Export corpus differs from the corpus used for attribution")
    selected = select_top_fraction(scores, fraction)
    if not len(selected):
        raise ValueError("Selection is empty; increase the fraction or input count")
    out = fresh_directory(output)
    rows = [{"id": records[i]["id"], "index": records[i].get("index", int(i)),
             "rank": rank, "score": float(scores[i])} for rank, i in enumerate(selected.tolist(), 1)]
    write_jsonl(out / "selected_ids.jsonl", rows)
    result = {"fraction": fraction, "candidates": len(scores), "selected": len(rows),
              "tie_break": "higher score first, then earlier feature row",
              "fit_mode": manifest["fit_mode"], "excluded_corpus_rows": manifest["excluded_corpus_rows"],
              "selected_ids_sha256": sha256_file(out / "selected_ids.jsonl")}
    if corpus:
        result["subset"] = export_subset(corpus, [row["id"] for row in rows], out / "subset.jsonl")
    write_json(out / "manifest.json", result)
    return result


def demo(output):
    """A CPU-only synthetic feature example, exercising real scoring and export."""
    from .scoring import task_centroids, score_features, debias_scores, select_top_fraction
    out = fresh_directory(output)
    rng = np.random.default_rng(42)
    training = rng.normal(size=(100, 16)) * np.exp(rng.normal(size=(100, 1)))
    validation = rng.normal(size=(12, 16))
    anchors = [{"id": f"v{i}", "task": "task_a" if i < 6 else "task_b",
                "query_id": f"q{i//3}", "reward": 1 if i % 3 else -1} for i in range(12)]
    centroids = task_centroids(validation, anchors)
    raw = score_features(training, centroids["centroids"])
    corrected = debias_scores(raw["raw"], raw["norms"])
    selected = select_top_fraction(corrected["scores"])
    np.save(out / "scores.npy", corrected["scores"], allow_pickle=False)
    write_json(out / "summary.json", {"synthetic": True, "candidates": 100, "anchors": 12,
        "selected_indices": selected.tolist(), "tasks": centroids["tasks"],
        "betas": corrected["betas"].tolist(), "fits": corrected["fits"]})
    return {"synthetic": True, "candidates": 100, "selected": len(selected), "output": str(out)}


def parser():
    p = argparse.ArgumentParser(prog="drift", description="On-policy influence scoring and instruction-data refinement")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("prepare", help="Normalize JSONL/Parquet while preserving all conversation turns")
    s.add_argument("--input", required=True); s.add_argument("--output", required=True)
    s = sub.add_parser("rollout", help="Generate model responses to canonical validation queries")
    s.add_argument("--queries", required=True); s.add_argument("--output", required=True)
    s.add_argument("--model", required=True); s.add_argument("--revision")
    s.add_argument("--samples", type=int, default=32); s.add_argument("--temperature", type=float, default=.6)
    s.add_argument("--top-p", type=float, default=.95); s.add_argument("--top-k", type=int, default=-1)
    s.add_argument("--max-new-tokens", type=int, default=2048); s.add_argument("--max-model-len", type=int)
    s.add_argument("--seed", type=int, default=42); s.add_argument("--rank", type=int, default=0)
    s.add_argument("--world-size", type=int, default=1)
    s.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    s = sub.add_parser("label", help="Verify replies and build reward-signed anchors")
    s.add_argument("--queries", required=True); s.add_argument("--rollouts", nargs="+", required=True)
    s.add_argument("--output", required=True); s.add_argument("--external-labels")
    s.add_argument("--allow-exclusions", action="store_true")
    s = sub.add_parser("extract", help="Extract assistant-token CE gradients and GraSS features")
    s.add_argument("--input", required=True); s.add_argument("--output", required=True)
    s.add_argument("--model", required=True); s.add_argument("--revision")
    s.add_argument("--dimension", type=int, default=16384); s.add_argument("--projection-seed", type=int, default=42)
    s.add_argument("--chunk-elements", type=int, default=536870911); s.add_argument("--max-length", type=int, default=4096)
    s.add_argument("--rank", type=int, default=0); s.add_argument("--world-size", type=int, default=1)
    s.add_argument("--device", default="cuda:0"); s.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    s.add_argument("--attention", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa")
    s.add_argument("--save-every", type=int, default=256); s.add_argument("--exclude-parameters")
    s.add_argument("--skip-no-supervision", action="store_true"); s.add_argument("--gradient-checkpointing", action="store_true")
    s = sub.add_parser("merge", help="Check rank/row coverage and merge feature shards")
    s.add_argument("--shards", nargs="+", required=True); s.add_argument("--output", required=True)
    s = sub.add_parser("score", help="Build task centroids, score corpus, and fit norm correction")
    s.add_argument("--corpus-features", required=True); s.add_argument("--anchor-features", required=True)
    s.add_argument("--output", required=True); s.add_argument("--fit-mode", choices=["stable", "legacy-float32"], default="stable")
    s.add_argument("--clip", type=float, nargs=2, metavar=("LOW_PERCENTILE", "HIGH_PERCENTILE"))
    s.add_argument("--zscore", action="store_true")
    s = sub.add_parser("select", help="Export the top fraction and optionally the complete conversation subset")
    s.add_argument("--scores-dir", required=True); s.add_argument("--output", required=True)
    s.add_argument("--fraction", type=float, default=.1); s.add_argument("--corpus")
    s = sub.add_parser("demo", help="Run a small CPU-only synthetic example")
    s.add_argument("--output", required=True)
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv); options = vars(args).copy(); command = options.pop("command")
    try:
        if command == "prepare":
            from .data import prepare_corpus
            result = prepare_corpus(options["input"], options["output"])
        elif command == "rollout":
            from .rollout import generate_rollouts
            options["queries_path"] = options.pop("queries"); options["output_dir"] = options.pop("output")
            result = generate_rollouts(**options)
        elif command == "label":
            from .rewards import label_rollouts
            result = label_rollouts(options["queries"], options["rollouts"], options["output"],
                options["external_labels"], allow_exclusions=options["allow_exclusions"])
        elif command == "extract":
            from .features import extract_features
            options["input_path"] = options.pop("input"); options["output_dir"] = options.pop("output")
            result = extract_features(**options)
        elif command == "merge":
            from .features import merge_features
            result = merge_features(options["shards"], options["output"])
        elif command == "score":
            result = run_score(**options)
        elif command == "select":
            result = run_select(**options)
        else:
            result = demo(**options)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, FileExistsError, ImportError, RuntimeError) as error:
        p.exit(2, f"drift: {error}\n")


if __name__ == "__main__":
    main()
