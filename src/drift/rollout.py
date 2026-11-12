"""Sample responses to validation queries from a frozen checkpoint.

Each rank owns queries whose input row index modulo world_size equals rank.
Launch ranks independently with the desired CUDA_VISIBLE_DEVICES; this module
does not assume a machine's GPU count or initialize a distributed process group.
GPU libraries are imported only when generation is requested.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .rewards import _load_queries, _sha256


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _request_seed(seed: int, query_id: str, sample_index: int) -> int:
    """Stable across rank count, batch size, and Python's hash randomization."""
    return int(_digest([seed, query_id, sample_index])[:16], 16) % (2**31)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _versions() -> dict[str, str]:
    result = {}
    for name in ("torch", "transformers", "tokenizers", "vllm"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def _eos_ids(config, generation_config, tokenizer) -> list[int]:
    # generation_config can specify multiple turn-ending tokens beyond the
    # tokenizer's eos_token_id. Do not replace them with a hardcoded sentinel.
    configured = generation_config.eos_token_id
    if configured is None:
        configured = getattr(config, "eos_token_id", None)
    if configured is None:
        configured = tokenizer.eos_token_id
    ids = configured if isinstance(configured, (list, tuple)) else [configured]
    ids = [int(value) for value in ids if value is not None]
    if not ids:
        raise ValueError("Model/tokenizer has no EOS token; supply a correctly configured chat model")
    return sorted(set(ids))


def _context_limit(config, tokenizer, requested: int | None) -> int:
    if requested is not None:
        return requested
    candidates = [getattr(config, key, None) for key in ("max_position_embeddings", "n_positions", "seq_length")]
    candidates.append(getattr(tokenizer, "model_max_length", None))
    finite = [int(value) for value in candidates if isinstance(value, (int, float)) and 0 < value < 10**8]
    if not finite:
        raise ValueError("Cannot determine context length; set max_model_len explicitly")
    return min(finite)


def _local_model_files(model: str) -> list[dict[str, Any]]:
    """Record local artifact metadata; do not claim weight-content hashing."""
    path = Path(model)
    if not path.is_dir():
        return []
    result = []
    for item in sorted(path.iterdir()):
        if item.is_file() and (item.suffix in {".json", ".safetensors", ".bin", ".model"}):
            entry = {"name": item.name, "bytes": item.stat().st_size}
            if item.suffix == ".json":
                entry["sha256"] = _sha256(item)
            result.append(entry)
    return result


def generate_rollouts(
    queries_path: str | Path,
    output_dir: str | Path,
    model: str,
    samples: int = 32,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = -1,
    max_new_tokens: int = 2048,
    max_model_len: int | None = None,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
    revision: str | None = None,
    backend: str = "vllm",
) -> dict[str, Any]:
    """Sample all completions into a new directory and return its manifest.

    Every sample is retained, including one-token answers, empty text, and
    length-limited outputs. Prompts are never silently truncated. A prompt plus
    max_new_tokens must fit max_model_len; otherwise generation fails with a
    diagnostic. The manifest and partial JSONL survive an interrupted run.

    vLLM uses offline batches of up to 128 independent, deterministically seeded
    requests. Transformers is a slower single-request fallback suitable for
    small models and integration checks. Seeds do not guarantee bitwise equality
    across hardware, library versions, or generation backends.
    """
    for name, value, minimum in (("samples", samples, 1), ("max_new_tokens", max_new_tokens, 1),
                                 ("world_size", world_size, 1), ("rank", rank, 0), ("seed", seed, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if rank >= world_size:
        raise ValueError("rank must be smaller than world_size")
    if max_model_len is not None and (type(max_model_len) is not int or max_model_len < 1):
        raise ValueError("max_model_len must be a positive integer")
    if (not isinstance(temperature, (int, float)) or not math.isfinite(temperature)
            or temperature < 0 or not 0 < top_p <= 1 or type(top_k) is not int or (top_k != -1 and top_k < 1)):
        raise ValueError("Require temperature >= 0, 0 < top_p <= 1, and top_k=-1 or a positive integer")
    if backend not in {"vllm", "transformers"}:
        raise ValueError("backend must be 'vllm' or 'transformers'")
    queries = _load_queries(queries_path)
    assigned = [row for index, row in enumerate(queries.values()) if index % world_size == rank]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    rollout_path = output_dir / "rollouts.jsonl"
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1, "status": "initializing", "started_at": _utc_now(),
        "model": model, "revision": revision, "backend": backend,
        "queries_path": str(queries_path), "queries_sha256": _sha256(queries_path),
        "rollout_file": str(rollout_path), "rank": rank, "world_size": world_size,
        "sharding": "input_row_index_modulo_world_size", "query_count": len(queries),
        "assigned_query_ids": [row["id"] for row in assigned], "generated_completions": 0,
        "expected_completions": len(assigned) * samples, "empty_completions": 0,
        "truncated_completions": 0, "versions": _versions(),
        "sampler": {"samples": samples, "temperature": temperature, "top_p": top_p,
                    "top_k": top_k, "max_new_tokens": max_new_tokens, "seed": seed,
                    "seed_derivation": "sha256_json([seed,query_id,sample_index])_first64bits_mod_2**31",
                    "ignore_eos": False, "min_tokens": 0, "repetition_penalty": 1.0},
        "prompt_truncation": False, "reference_answers_injected": False,
    }

    def save_manifest() -> None:
        temporary = output_dir / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)

    save_manifest()
    try:
        from transformers import AutoConfig, AutoTokenizer, GenerationConfig

        config = AutoConfig.from_pretrained(model, revision=revision, trust_remote_code=False)
        resolved_revision = getattr(config, "_commit_hash", None) or revision
        tokenizer = AutoTokenizer.from_pretrained(model, revision=resolved_revision, trust_remote_code=False)
        try:
            generation_config = GenerationConfig.from_pretrained(model, revision=resolved_revision)
        except OSError:
            generation_config = GenerationConfig.from_model_config(config)
        template = tokenizer.get_chat_template()
        if not template:
            raise ValueError("Tokenizer has no chat template; provide a chat/SFT tokenizer")
        eos_ids = _eos_ids(config, generation_config, tokenizer)
        context_limit = _context_limit(config, tokenizer, max_model_len)
        tokenizer_dir = output_dir / "tokenizer"
        tokenizer.save_pretrained(tokenizer_dir)
        config.save_pretrained(output_dir / "model_config")
        manifest.update({
            "resolved_model_revision": getattr(config, "_commit_hash", None),
            "model_config_sha256": _digest(config.to_dict()),
            "local_model_artifacts": _local_model_files(model),
            "weight_identity": "hub_revision_when_resolved; local_weight_names_and_sizes_only",
            "tokenizer": {"name_or_path": tokenizer.name_or_path,
                          "resolved_revision": tokenizer.init_kwargs.get("_commit_hash"),
                          "files": {item.name: _sha256(item) for item in sorted(tokenizer_dir.iterdir()) if item.is_file()},
                          "chat_template": template, "chat_template_sha256": _digest(template),
                          "add_generation_prompt": True, "eos_token_ids": eos_ids},
            "max_model_len": context_limit,
        })
        manifest["sampler"]["stop_token_ids"] = eos_ids
        prompts = []
        for query in assigned:
            ids = tokenizer.apply_chat_template(query["messages"], tokenize=True, add_generation_prompt=True)
            if not isinstance(ids, list) or not all(isinstance(token, int) for token in ids):
                raise ValueError(f"Query {query['id']}: chat template did not return token IDs")
            if len(ids) + max_new_tokens > context_limit:
                raise ValueError(
                    f"Query {query['id']}: {len(ids)} prompt tokens + {max_new_tokens} generation tokens "
                    f"exceed max_model_len={context_limit}; increase context or revise the input/budget explicitly"
                )
            prompts.append((query, ids))
        manifest["status"] = "running"
        save_manifest()

        def write_completion(handle, query, prompt_ids, sample_index, completion, token_ids, finish_reason, stop_reason=None):
            row = {
                "id": f"{query['id']}:{sample_index}", "query_id": query["id"], "task": query["task"],
                "sample_index": sample_index, "seed": _request_seed(seed, query["id"], sample_index),
                "messages": query["messages"] + [{"role": "assistant", "content": completion}],
                "completion": completion, "finish_reason": finish_reason, "stop_reason": stop_reason,
                "generated_tokens": len(token_ids), "completion_token_ids": list(token_ids),
                "prompt_tokens": len(prompt_ids), "prompt_token_ids_sha256": _digest(prompt_ids),
                "truncated": finish_reason == "length", "empty": not completion.strip(),
                "rank": rank,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            manifest["generated_completions"] += 1
            manifest["empty_completions"] += int(row["empty"])
            manifest["truncated_completions"] += int(row["truncated"])

        with rollout_path.open("x", encoding="utf-8") as handle:
            if prompts and backend == "vllm":
                from vllm import LLM, SamplingParams

                engine = LLM(model=model, revision=resolved_revision, tokenizer=model, tokenizer_revision=resolved_revision,
                             trust_remote_code=False, tensor_parallel_size=1, max_model_len=context_limit,
                             seed=seed, generation_config="vllm")
                requests = [(query, ids, sample_index) for query, ids in prompts for sample_index in range(samples)]
                manifest["offline_batch_size"] = 128
                for offset in range(0, len(requests), 128):
                    batch = requests[offset:offset + 128]
                    parameters = [SamplingParams(n=1, temperature=temperature, top_p=top_p, top_k=top_k,
                                                 max_tokens=max_new_tokens, min_tokens=0, stop_token_ids=eos_ids,
                                                 ignore_eos=False, repetition_penalty=1.0,
                                                 seed=_request_seed(seed, query["id"], sample_index))
                                  for query, _, sample_index in batch]
                    outputs = engine.generate([{"prompt_token_ids": ids} for _, ids, _ in batch],
                                              sampling_params=parameters, use_tqdm=False)
                    if len(outputs) != len(batch):
                        raise RuntimeError("vLLM returned an unexpected number of requests")
                    for (query, ids, sample_index), output in zip(batch, outputs):
                        if len(output.outputs) != 1:
                            raise RuntimeError("vLLM returned an unexpected number of completions")
                        completion = output.outputs[0]
                        if completion.finish_reason not in {"stop", "length"}:
                            raise RuntimeError(f"Unfinished vLLM request: {query['id']}:{sample_index}")
                        write_completion(handle, query, ids, sample_index, completion.text,
                                         completion.token_ids, completion.finish_reason, completion.stop_reason)
                    save_manifest()
            elif prompts:
                import torch
                from transformers import AutoModelForCausalLM

                device = "cuda" if torch.cuda.is_available() else "cpu"
                language_model = AutoModelForCausalLM.from_pretrained(model, revision=resolved_revision,
                                                                    torch_dtype="auto", trust_remote_code=False)
                language_model.to(device).eval()
                manifest["device"] = device
                for query, ids in prompts:
                    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                    for sample_index in range(samples):
                        torch.manual_seed(_request_seed(seed, query["id"], sample_index))
                        options = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
                                   "eos_token_id": eos_ids, "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0],
                                   "use_cache": True, "repetition_penalty": 1.0}
                        if temperature > 0:
                            options.update(temperature=temperature, top_p=top_p, top_k=0 if top_k == -1 else top_k)
                        with torch.inference_mode():
                            generated = language_model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                                                generation_config=GenerationConfig(**options))
                        token_ids = generated[0, len(ids):].tolist()
                        stopped = bool(token_ids) and token_ids[-1] in eos_ids
                        completion = tokenizer.decode(token_ids, skip_special_tokens=True)
                        write_completion(handle, query, ids, sample_index, completion, token_ids,
                                         "stop" if stopped else "length", token_ids[-1] if stopped else None)
                    save_manifest()
        if manifest["generated_completions"] != manifest["expected_completions"]:
            raise RuntimeError("Incomplete rollout count")
        if _sha256(queries_path) != manifest["queries_sha256"]:
            raise RuntimeError("Queries changed while generation was running")
        manifest["status"] = "complete"
        manifest["rollouts_sha256"] = _sha256(rollout_path)
        manifest["finished_at"] = _utc_now()
    except BaseException as exc:
        manifest["status"] = "failed" if isinstance(exc, Exception) else "interrupted"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["finished_at"] = _utc_now()
        raise
    finally:
        save_manifest()
    return manifest
