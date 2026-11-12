"""Extract, merge, and validate projected per-example CE gradients."""
from __future__ import annotations

import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np

from .io import (canonical_json, fresh_directory, object_hash, read_jsonl,
                 require_id, sha256_file, write_json, write_jsonl)
from .masking import assistant_labels
from .projection import GrassProjector


def _model_identity(model_path, config):
    path = Path(model_path)
    identity = {"config": {k: v for k, v in config.to_dict().items()
                           if k not in {"_name_or_path", "transformers_version"}}}
    if path.is_dir():
        files = sorted(path.glob("*.safetensors")) or sorted(path.glob("pytorch_model*.bin"))
        if not files:
            raise ValueError("Cannot fingerprint local model weights")
        identity["weights"] = {p.name: sha256_file(p) for p in files}
    else:
        revision = getattr(config, "_commit_hash", None)
        if not revision:
            raise ValueError("Remote model has no resolved commit; use a pinned revision or local snapshot")
        identity["repo"] = str(model_path)
        identity["resolved_revision"] = revision
    return identity


def extract_features(input_path, output_dir, model, *, revision=None, dimension=16384,
                     projection_seed=42, chunk_elements=536870911, max_length=4096,
                     rank=0, world_size=1, device="cuda:0", dtype="bfloat16",
                     attention="sdpa", save_every=256, exclude_parameters=None,
                     skip_no_supervision=False, gradient_checkpointing=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if world_size < 1 or not 0 <= rank < world_size or save_every < 1:
        raise ValueError("Invalid rank, world size or chunk size")
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("Unsupported model dtype")
    out = fresh_directory(output_dir)
    (out / "chunks").mkdir()
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    if not tokenizer.is_fast or not tokenizer.chat_template:
        raise ValueError("A fast tokenizer with an explicit chat template is required")
    network = AutoModelForCausalLM.from_pretrained(model, revision=revision,
        torch_dtype=getattr(torch, dtype), attn_implementation=attention, trust_remote_code=False)
    network.to(device)
    network.eval()
    if gradient_checkpointing:
        def nonzero_dropout(value):
            if isinstance(value, dict):
                return any(("drop" in key.lower() and isinstance(child, (int, float))
                            and not isinstance(child, bool) and child > 0)
                           or nonzero_dropout(child) for key, child in value.items())
            return False
        if nonzero_dropout(network.config.to_dict()):
            raise ValueError("Checkpointed extraction requires a zero-dropout model configuration; use eval-mode extraction otherwise")
        network.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Transformers activates checkpointing only in train mode. Turn off all
        # dropout modules explicitly so attribution remains deterministic.
        network.train()
        for module in network.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    pattern = re.compile(exclude_parameters) if exclude_parameters else None
    for name, parameter in network.named_parameters():
        if pattern and pattern.search(name):
            parameter.requires_grad_(False)
    parameters = sorted((name, p) for name, p in network.named_parameters() if p.requires_grad)
    projector = GrassProjector([(name, tuple(p.shape)) for name, p in parameters],
                                dimension, projection_seed, chunk_elements)
    projector.save(out)
    identity = {"model": _model_identity(model, network.config),
                "tokenizer_vocab": object_hash(tokenizer.get_vocab()),
                "tokenizer_backend": object_hash(tokenizer.backend_tokenizer.to_str()),
                "tokenizer_behavior": {"truncation_side": tokenizer.truncation_side,
                                       "padding_side": tokenizer.padding_side,
                                       "special_tokens": tokenizer.special_tokens_map},
                "chat_template": object_hash(tokenizer.chat_template),
                "projection": projector.fingerprint, "max_length": max_length,
                "mask": "assistant-content-eos-v3:corpus-all-turns:anchor-final-turn-complete", "loss": "causal-token-mean-ce",
                "dtype": dtype, "attention": attention, "exclude_parameters": exclude_parameters,
                "gradient_checkpointing": gradient_checkpointing,
                "versions": {n: importlib.metadata.version(n) for n in ["torch", "transformers", "numpy"]}}
    manifest = {"schema_version": 1, "state": "running", "rank": rank, "world_size": world_size,
                "input_sha256": sha256_file(input_path), "identity": identity,
                "feature_fingerprint": object_hash(identity), "dimension": dimension,
                "chunks": [], "skip_no_supervision": skip_no_supervision,
                "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(out / "manifest.json", manifest)
    seen = set()
    features, indices, records, rejected = [], [], [], []
    nrows = 0
    assigned = 0
    completed = 0
    started = time.monotonic()

    def checkpoint():
        nonlocal features, indices, records
        if not features:
            return
        stem = f"{len(manifest['chunks']):06d}"
        array_path, metadata_path = out / "chunks" / f"{stem}.npz", out / "chunks" / f"{stem}.jsonl"
        with array_path.open("xb") as stream:
            np.savez(stream, features=np.stack(features), indices=np.asarray(indices, dtype=np.int64))
        write_jsonl(metadata_path, records)
        manifest["chunks"].append({"array": str(array_path.relative_to(out)),
            "metadata": str(metadata_path.relative_to(out)), "rows": len(records),
            "array_sha256": sha256_file(array_path), "metadata_sha256": sha256_file(metadata_path)})
        temp = out / "manifest.tmp.json"
        write_json(temp, manifest)
        os.replace(temp, out / "manifest.json")
        features, indices, records = [], [], []

    for index, row in enumerate(read_jsonl(input_path)):
        identifier = require_id(row)
        if identifier in seen:
            raise ValueError(f"Duplicate input ID: {identifier}")
        seen.add(identifier)
        nrows += 1
        if index % world_size != rank:
            continue
        assigned += 1
        encoded = assistant_labels(tokenizer, row["messages"], max_length,
                                   last_only="reward" in row, require_complete="reward" in row)
        if not encoded["supervised_tokens"]:
            if not skip_no_supervision or "reward" in row:
                raise ValueError(f"No assistant targets after truncation: {identifier}")
            rejected.append({"id": identifier, "index": index, "reason": "zero_supervision_after_truncation"})
            continue
        network.zero_grad(set_to_none=True)
        ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
        labels = torch.tensor([encoded["labels"]], dtype=torch.long, device=device)
        loss = network(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss: {identifier}")
        loss_value = float(loss.detach())
        loss.backward()
        vector = projector.project_parameters(parameters).detach().cpu().numpy()
        if not np.isfinite(vector).all():
            raise ValueError(f"Non-finite projected gradient: {identifier}")
        metadata = {k: v for k, v in row.items() if k != "messages"}
        metadata.update({"index": index, "record_sha256": object_hash(row),
                         "tokens": len(encoded["input_ids"]),
                         "supervised_tokens": encoded["supervised_tokens"], "loss": loss_value})
        features.append(vector)
        indices.append(index)
        records.append(metadata)
        completed += 1
        del loss, ids, labels
        if len(features) >= save_every:
            checkpoint()
            print(canonical_json({"rank": rank, "completed": completed,
                                  "elapsed_seconds": time.monotonic()-started}), flush=True)
    checkpoint()
    if not nrows:
        raise ValueError("Empty input")
    if sha256_file(input_path) != manifest["input_sha256"]:
        raise ValueError("Input changed during gradient extraction")
    write_jsonl(out / "rejected.jsonl", rejected)
    manifest.update({"state": "complete", "input_rows": nrows, "assigned_rows": assigned,
                     "feature_rows": completed, "rejected_rows": len(rejected),
                     "rejected_sha256": sha256_file(out / "rejected.jsonl"),
                     "elapsed_seconds": time.monotonic()-started})
    write_json(out / "manifest.json", manifest, exclusive=False)
    return manifest


def merge_features(shard_dirs, output_dir):
    """Validate rank coverage and exact row identity before writing a feature table."""
    roots = [Path(p) for p in shard_dirs]
    if not roots:
        raise ValueError("No feature shards")
    manifests = [json.loads((p / "manifest.json").read_text()) for p in roots]
    first = manifests[0]
    for manifest in manifests:
        if manifest.get("state") != "complete":
            raise ValueError("Incomplete extraction shard")
        for key in ("world_size", "input_sha256", "input_rows", "feature_fingerprint", "dimension", "skip_no_supervision"):
            if manifest[key] != first[key]:
                raise ValueError(f"Incompatible feature shards: {key}")
    if sorted(m["rank"] for m in manifests) != list(range(first["world_size"])):
        raise ValueError("Missing or duplicated ranks")
    count = sum(m["feature_rows"] for m in manifests)
    if not count:
        raise ValueError("No valid features")
    out = fresh_directory(output_dir)
    records = {}
    excluded = {}
    ids = set()
    for root, manifest in zip(roots, manifests):
        if sha256_file(root / "rejected.jsonl") != manifest["rejected_sha256"]:
            raise ValueError("Rejection ledger checksum mismatch")
        rejected = list(read_jsonl(root / "rejected.jsonl"))
        if len(rejected) != manifest["rejected_rows"]:
            raise ValueError("Rejection count differs from manifest")
        for row in rejected:
            index = row["index"]
            if index in excluded or index < 0 or index >= first["input_rows"] or index % first["world_size"] != manifest["rank"]:
                raise ValueError("Invalid rejection row identity")
            excluded[index] = row
    if count + len(excluded) != first["input_rows"]:
        raise ValueError("Feature and rejection counts do not cover the input")
    needed_bytes = count * first["dimension"] * np.dtype(np.float32).itemsize
    if shutil.disk_usage(out).free < needed_bytes + 64*1024**2:
        raise OSError(f"Insufficient output space: at least {needed_bytes} feature bytes are needed")
    # Map original input positions directly into the final disk-backed table.
    # This avoids a second full N-by-d scratch matrix during merging.
    destination = np.full(first["input_rows"], -1, dtype=np.int64)
    valid_indices = np.asarray([i for i in range(first["input_rows"]) if i not in excluded], dtype=np.int64)
    destination[valid_indices] = np.arange(count, dtype=np.int64)
    matrix_out = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype=np.float32,
                                          shape=(count, first["dimension"]))
    for root, manifest in zip(roots, manifests):
        valid_for_rank = 0
        for chunk in manifest["chunks"]:
            array_path, meta_path = root / chunk["array"], root / chunk["metadata"]
            if sha256_file(array_path) != chunk["array_sha256"] or sha256_file(meta_path) != chunk["metadata_sha256"]:
                raise ValueError("Feature chunk checksum mismatch")
            with np.load(array_path, allow_pickle=False) as values:
                matrix, indices = values["features"], values["indices"]
                metadata = list(read_jsonl(meta_path))
                if matrix.shape != (chunk["rows"], first["dimension"]) or len(indices) != len(metadata) or len(metadata) != chunk["rows"]:
                    raise ValueError("Feature/metadata dimensions disagree")
                if not np.isfinite(matrix).all():
                    raise ValueError("Non-finite feature block")
                for j, (index, row) in enumerate(zip(indices.tolist(), metadata)):
                    if index < 0 or index >= first["input_rows"] or index % first["world_size"] != manifest["rank"]:
                        raise ValueError("Feature row belongs to the wrong rank or input")
                    if index != row["index"] or index in records or index in excluded or row["id"] in ids:
                        raise ValueError("Duplicated or mismatched feature row identity")
                    records[index] = row
                    ids.add(row["id"])
                    matrix_out[destination[index]] = matrix[j]
                    valid_for_rank += 1
        if valid_for_rank != manifest["feature_rows"]:
            raise ValueError("Shard feature count differs from manifest")
    if set(records) & set(excluded) or set(records) | set(excluded) != set(range(first["input_rows"])):
        raise ValueError("Missing or duplicate rows; refusing to fill missing gradients with zero")
    ordered = sorted(records)
    matrix_out.flush()
    del matrix_out
    write_jsonl(out / "records.jsonl", (records[i] for i in ordered))
    write_jsonl(out / "rejected.jsonl", (excluded[i] for i in sorted(excluded)))
    result = {"schema_version": 1, "state": "complete", "rows": count,
              "input_rows": first["input_rows"], "rejected_rows": len(excluded),
              "input_sha256": first["input_sha256"], "dimension": first["dimension"],
              "identity": first["identity"], "feature_fingerprint": first["feature_fingerprint"],
              "features_sha256": sha256_file(out / "features.npy"),
              "records_sha256": sha256_file(out / "records.jsonl")}
    write_json(out / "manifest.json", result)
    return result


def load_feature_table(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("state") != "complete":
        raise ValueError("Incomplete feature table")
    if sha256_file(root / "features.npy") != manifest["features_sha256"] or sha256_file(root / "records.jsonl") != manifest["records_sha256"]:
        raise ValueError("Feature table checksum mismatch")
    matrix = np.load(root / "features.npy", mmap_mode="r", allow_pickle=False)
    records = list(read_jsonl(root / "records.jsonl"))
    if matrix.shape != (manifest["rows"], manifest["dimension"]) or len(records) != manifest["rows"]:
        raise ValueError("Feature table dimensions disagree")
    return matrix, records, manifest
