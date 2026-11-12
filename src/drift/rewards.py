"""Verify sampled completions and build signed DRIFT validation anchors.

Reference answers are used only by verifiers: they are never added as positive
training examples. Generated code is never executed here. Use the ``external``
verifier to import results from the benchmark's own isolated evaluation harness.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield row


def _load_queries(path: str | Path) -> dict[str, dict[str, Any]]:
    queries: dict[str, dict[str, Any]] = {}
    for row in _jsonl(path):
        query_id = row.get("id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("Every query needs a nonempty string id")
        if query_id in queries:
            raise ValueError(f"Duplicate query id: {query_id}")
        if not isinstance(row.get("task"), str) or not row["task"]:
            raise ValueError(f"Query {query_id}: missing task")
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Query {query_id}: messages must be a nonempty list")
        for message in messages:
            if (not isinstance(message, dict)
                    or message.get("role") not in {"system", "user", "assistant"}
                    or not isinstance(message.get("content"), str)):
                raise ValueError(f"Query {query_id}: invalid message")
        if messages[-1]["role"] != "user":
            raise ValueError(f"Query {query_id}: the final prompt message must be user")
        verifier = row.get("verifier")
        if not isinstance(verifier, dict) or verifier.get("type") not in {
            "choice", "exact", "math", "external"
        }:
            raise ValueError(f"Query {query_id}: unsupported or missing verifier")
        if verifier["type"] != "external" and "answer" not in verifier:
            raise ValueError(f"Query {query_id}: verifier.answer is required")
        if verifier["type"] == "choice":
            choices = verifier.get("choices")
            if choices is not None and (not isinstance(choices, list) or not choices):
                raise ValueError(f"Query {query_id}: choices must be a nonempty list")
            if _canonical_choice(verifier["answer"], choices) is None:
                raise ValueError(f"Query {query_id}: answer must be a valid choice label")
        if verifier["type"] == "exact":
            normalize_exact("", verifier.get("normalization"))
            if not isinstance(verifier["answer"], str):
                raise ValueError(f"Query {query_id}: exact answer must be a string")
            pattern = verifier.get("extract_regex")
            if pattern is not None:
                if not isinstance(pattern, str):
                    raise ValueError(f"Query {query_id}: extract_regex must be a string")
                try:
                    compiled = re.compile(pattern)
                except re.error as exc:
                    raise ValueError(f"Query {query_id}: invalid extract_regex") from exc
                if compiled.groups != 1:
                    raise ValueError(f"Query {query_id}: extract_regex must have exactly one capture group")
        queries[query_id] = row
    if not queries:
        raise ValueError("The validation query file is empty")
    return queries


def _choice_label(index: int) -> str:
    result = ""
    while index >= 0:
        index, digit = divmod(index, 26)
        result = chr(65 + digit) + result
        index -= 1
    return result


def _canonical_choice(value: Any, choices: list[Any] | None) -> str | None:
    if not isinstance(value, str):
        return None
    label = value.strip().strip("()[]{}").strip().upper()
    valid = {_choice_label(i) for i in range(len(choices) if choices else 26)}
    return label if label in valid else None


def parse_choice(completion: str, choices: list[Any] | None = None) -> str | None:
    """Extract a label without access to the reference answer.

    Explicit answer markers outrank a bare answer; contradictory explicit
    markers are treated as ambiguous. Labels are A..Z, AA.. when choices has
    more than 26 entries. We deliberately do not search arbitrary body text for
    a letter that happens to match the reference.
    """
    marked: list[str] = []
    patterns = (
        r"\\boxed\s*\{\s*([A-Za-z]{1,3})\s*\}",
        r"\b(?:final\s+answer|answer)\s*(?:is\b|[:=])\s*[*_\s]*[([{]?\s*([A-Za-z]{1,3})\b",
    )
    for pattern in patterns:
        marked.extend(re.findall(pattern, completion, flags=re.IGNORECASE))
    if marked:
        labels = [_canonical_choice(value, choices) for value in marked]
        return labels[0] if labels[0] is not None and len(set(labels)) == 1 else None
    bare = re.fullmatch(r"\s*[*_]*\s*[([{]?\s*([A-Za-z]{1,3})\s*[)\]}]?[.!]?\s*[*_]*\s*", completion)
    if bare:
        return _canonical_choice(bare.group(1), choices)
    # A complete single-line choice such as "J. N/A" or "H. 100".
    # Never inspect the reference answer or match letters within reasoning.
    prefixed = re.fullmatch(r"\s*([A-Za-z]{1,3})[.)]\s+[^\r\n]+\s*", completion)
    return _canonical_choice(prefixed.group(1), choices) if prefixed else None


DEFAULT_NORMALIZATION = {
    "unicode": "NFKC", "strip": True, "collapse_whitespace": True, "casefold": False,
}


def normalize_exact(text: str, normalization: dict[str, Any] | None = None) -> str:
    """NFKC + strip + whitespace collapse by default; case is significant."""
    if normalization is not None and not isinstance(normalization, dict):
        raise ValueError("exact normalization must be an object")
    options = {**DEFAULT_NORMALIZATION, **(normalization or {})}
    if set(options) != set(DEFAULT_NORMALIZATION):
        raise ValueError("Unknown exact normalization option")
    if options["unicode"] not in {None, "NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError("Unsupported Unicode normalization")
    for key in ("strip", "collapse_whitespace", "casefold"):
        if not isinstance(options[key], bool):
            raise ValueError(f"normalization.{key} must be boolean")
    if options["unicode"]:
        text = unicodedata.normalize(options["unicode"], text)
    if options["strip"]:
        text = text.strip()
    if options["collapse_whitespace"]:
        text = re.sub(r"\s+", " ", text)
    if options["casefold"]:
        text = text.casefold()
    return text


def _validate_rollout(row: dict[str, Any], queries: dict[str, dict[str, Any]]) -> None:
    query_id = row.get("query_id")
    if not isinstance(query_id, str) or query_id not in queries:
        raise ValueError(f"Unknown rollout query_id: {query_id!r}")
    query = queries[query_id]
    if row.get("task") != query["task"]:
        raise ValueError(f"Task mismatch for rollout {row.get('id')!r}")
    row_id = row.get("id")
    prefix = query_id + ":"
    if (not isinstance(row_id, str) or not row_id.startswith(prefix)
            or not row_id[len(prefix):].isdigit()
            or str(int(row_id[len(prefix):])) != row_id[len(prefix):]):
        raise ValueError(f"Rollout id must be query_id:sample_index: {row_id!r}")
    if "sample_index" in row and row["sample_index"] != int(row_id[len(prefix):]):
        raise ValueError(f"sample_index mismatch for {row_id}")
    completion = row.get("completion")
    messages = row.get("messages")
    if not isinstance(completion, str):
        raise ValueError(f"Rollout {row_id}: completion must be a string")
    if (not isinstance(messages, list) or messages[:-1] != query["messages"]
            or not messages or messages[-1] != {"role": "assistant", "content": completion}):
        raise ValueError(f"Rollout {row_id}: messages must preserve the complete query and completion")
    if row.get("finish_reason") not in {"stop", "length", "eos", "stop_sequence", "error", "abort"}:
        raise ValueError(f"Rollout {row_id}: missing or unsupported finish_reason")
    if "truncated" in row and not isinstance(row["truncated"], bool):
        raise ValueError(f"Rollout {row_id}: truncated must be boolean")


def _load_math_verifier():
    try:
        from math_verify import parse, verify
    except ImportError as exc:
        raise RuntimeError(
            "Math verification requires the optional math-verify package. "
            "Alternatively use verifier.type='external' and import official evaluator labels."
        ) from exc
    return parse, verify


def label_rollouts(
    queries_path: str | Path,
    rollout_paths: list[str | Path],
    output_path: str | Path,
    external_labels: str | Path | None = None,
    *,
    allow_exclusions: bool = False,
) -> dict[str, Any]:
    """Write signed anchors and ``<output_path>.audit.json`` without overwriting.

    All queries must be covered by the supplied rollout shards. External labels
    must cover exactly the externally verified rollout IDs, including excluded
    empty/truncated records. Empty, interrupted, truncated, and unverifiable
    math outputs fail the operation by default. ``allow_exclusions=True``
    explicitly permits excluding them, including queries left with no anchors;
    all excluded IDs and zero-anchor queries are recorded in the audit. They
    are never made negative merely because verification could not run.
    Normal empty parses from a nonempty choice/math/exact-regex answer are
    incorrect under the verifier's fixed extraction rule, and are counted as
    parse failures. Verifier exceptions are technical failures, not negatives.
    """
    queries = _load_queries(queries_path)
    if not rollout_paths:
        raise ValueError("At least one rollout path is required")
    output_path = Path(output_path)
    partial_path = Path(str(output_path) + ".partial")
    audit_path = Path(str(output_path) + ".audit.json")
    if output_path.exists() or audit_path.exists() or partial_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path} or its audit")
    seen: set[str] = set()
    external_ids: set[str] = set()
    coverage: Counter[str] = Counter()
    sample_indices: dict[str, set[int]] = {}
    for path in rollout_paths:
        manifest_path = Path(path).parent / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            # Only recognize manifests describing this exact file, rather than
            # unrelated experiment metadata in the same directory.
            if Path(manifest.get("rollout_file", "")).name == Path(path).name:
                if manifest.get("status") != "complete":
                    raise ValueError(f"Rollout manifest is not complete: {manifest_path}")
                if manifest.get("queries_sha256") != _sha256(queries_path):
                    raise ValueError(f"Rollout manifest references different queries: {manifest_path}")
                if manifest.get("rollouts_sha256") != _sha256(path):
                    raise ValueError(f"Rollout hash disagrees with manifest: {path}")
        for row in _jsonl(path):
            _validate_rollout(row, queries)
            if row["id"] in seen:
                raise ValueError(f"Duplicate rollout id: {row['id']}")
            seen.add(row["id"])
            coverage[row["query_id"]] += 1
            sample_indices.setdefault(row["query_id"], set()).add(int(row["id"].rsplit(":", 1)[1]))
            if queries[row["query_id"]]["verifier"]["type"] == "external":
                external_ids.add(row["id"])
    missing = set(queries) - set(coverage)
    if missing:
        raise ValueError(f"Missing rollout coverage for {len(missing)} queries; first: {sorted(missing)[:3]}")
    if any(indices != set(range(len(indices))) for indices in sample_indices.values()):
        raise ValueError("Rollout sample indices must be contiguous from zero for every query")
    if len(set(coverage.values())) != 1:
        raise ValueError("Unequal samples per query; supply all complete rollout shards")
    labels: dict[str, int] = {}
    if external_labels is not None:
        for row in _jsonl(external_labels):
            row_id, reward = row.get("id"), row.get("reward")
            if not isinstance(row_id, str) or row_id in labels:
                raise ValueError(f"Invalid or duplicate external label id: {row_id!r}")
            if type(reward) not in {int, float} or reward not in {-1, 1}:
                raise ValueError(f"External reward must be exactly -1 or +1: {row_id}")
            expected_query_id = row_id.rsplit(":", 1)[0]
            if "query_id" in row and row["query_id"] != expected_query_id:
                raise ValueError(f"External label query_id mismatch: {row_id}")
            if "task" in row and (expected_query_id not in queries or row["task"] != queries[expected_query_id]["task"]):
                raise ValueError(f"External label task mismatch: {row_id}")
            labels[row_id] = int(reward)
    if set(labels) != external_ids:
        raise ValueError(
            f"External labels do not exactly cover external rollouts: "
            f"missing={len(external_ids - set(labels))}, extra={len(set(labels) - external_ids)}"
        )
    math_api = _load_math_verifier() if any(q["verifier"]["type"] == "math" for q in queries.values()) else None
    math_gold: dict[str, Any] = {}
    if math_api:
        for query_id, query in queries.items():
            if query["verifier"]["type"] == "math":
                try:
                    value = math_api[0](str(query["verifier"]["answer"]), fallback_mode="no_fallback")
                except Exception as exc:
                    raise ValueError(f"Math reference is unverifiable for {query_id}; use external labels") from exc
                if not value:
                    raise ValueError(f"Math reference is unverifiable for {query_id}; use external labels")
                math_gold[query_id] = value
    audit: dict[str, Any] = {
        "schema_version": 1, "status": "running", "output_path": str(output_path),
        "queries_sha256": _sha256(queries_path),
        "rollouts": [{"path": str(p), "sha256": _sha256(p)} for p in rollout_paths],
        "external_labels_sha256": _sha256(external_labels) if external_labels is not None else None,
        "input_completions": len(seen), "output_anchors": 0,
        "positive": 0, "negative": 0, "excluded": [], "parse_failures": [], "tasks": {},
        "query_sample_counts": dict(coverage), "normalization_defaults": DEFAULT_NORMALIZATION,
        "choice_parser": "explicit_unambiguous_marker_or_bare_label_or_single_line_prefix_v1",
        "reference_answers_injected": False,
        "allow_exclusions": allow_exclusions,
    }
    if math_api:
        try:
            audit["math_verify_version"] = importlib.metadata.version("math-verify")
        except importlib.metadata.PackageNotFoundError:
            audit["math_verify_version"] = "unknown"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve both names before writing. An interrupted run remains visibly
    # incomplete and cannot silently replace an earlier anchor collection.
    with audit_path.open("x", encoding="utf-8") as audit_handle:
        json.dump(audit, audit_handle, ensure_ascii=False, indent=2)
    try:
        anchor_coverage: Counter[str] = Counter()
        with partial_path.open("x", encoding="utf-8") as handle:
            for path in rollout_paths:
                for row in _jsonl(path):
                    query = queries[row["query_id"]]
                    verifier = query["verifier"]
                    text = row["completion"]
                    excluded = None
                    if not text.strip():
                        excluded = "empty_completion"
                    elif row.get("truncated") or row["finish_reason"] == "length":
                        excluded = "truncated_completion"
                    elif row["finish_reason"] in {"error", "abort"}:
                        excluded = "generation_error"
                    reward = None
                    parse_failure = None
                    if excluded is None and verifier["type"] == "external":
                        reward = labels[row["id"]]
                    elif excluded is None and verifier["type"] == "choice":
                        prediction = parse_choice(text, verifier.get("choices"))
                        if prediction is None:
                            parse_failure = "choice_no_unambiguous_label"
                        reward = 1 if prediction == _canonical_choice(verifier["answer"], verifier.get("choices")) else -1
                    elif excluded is None and verifier["type"] == "exact":
                        options = verifier.get("normalization")
                        pattern = verifier.get("extract_regex")
                        extracted = re.findall(pattern, text) if pattern is not None else [text]
                        if len(extracted) != 1:
                            parse_failure = "exact_regex_missing_or_ambiguous"
                            reward = -1
                        else:
                            reward = 1 if normalize_exact(extracted[0], options) == normalize_exact(verifier["answer"], options) else -1
                    elif excluded is None and verifier["type"] == "math":
                        try:
                            prediction = math_api[0](text, fallback_mode="no_fallback")
                            if not prediction:
                                parse_failure = "math_no_parsed_answer"
                                reward = -1
                            else:
                                result = math_api[1](math_gold[row["query_id"]], prediction)
                                if not isinstance(result, bool):
                                    excluded = "math_unverifiable_result"
                                else:
                                    reward = 1 if result else -1
                        except Exception as exc:
                            excluded = "math_unverifiable_" + type(exc).__name__
                    task_counts = audit["tasks"].setdefault(row["task"], {"positive": 0, "negative": 0, "excluded": 0})
                    if parse_failure is not None:
                        audit["parse_failures"].append({"id": row["id"], "query_id": row["query_id"], "task": row["task"], "reason": parse_failure})
                    if excluded is not None:
                        audit["excluded"].append({"id": row["id"], "query_id": row["query_id"], "task": row["task"], "reason": excluded})
                        task_counts["excluded"] += 1
                        continue
                    anchor = {key: row[key] for key in ("id", "task", "query_id", "messages")}
                    anchor["reward"] = reward
                    handle.write(json.dumps(anchor, ensure_ascii=False) + "\n")
                    sign = "positive" if reward == 1 else "negative"
                    audit[sign] += 1
                    task_counts[sign] += 1
                    audit["output_anchors"] += 1
                    anchor_coverage[row["query_id"]] += 1
        # Detect files changing between validation and writing, including a
        # still-running generator. Such anchors must not be used downstream.
        if audit["queries_sha256"] != _sha256(queries_path) or any(
            source["sha256"] != _sha256(source["path"]) for source in audit["rollouts"]
        ):
            raise RuntimeError("Input changed during labeling; this output is incomplete")
        if external_labels is not None and audit["external_labels_sha256"] != _sha256(external_labels):
            raise RuntimeError("External labels changed during labeling")
        audit["zero_anchor_query_ids"] = sorted(set(queries) - set(anchor_coverage))
        audit["query_anchor_counts"] = {key: anchor_coverage[key] for key in queries}
        if audit["excluded"] and not allow_exclusions:
            raise ValueError(
                f"{len(audit['excluded'])} completions are empty, truncated, interrupted, or unverifiable; "
                "see audit. Import external verification for unverifiable answers, or explicitly "
                "set allow_exclusions=True to accept a changed anchor population."
            )
        if not audit["output_anchors"]:
            raise ValueError("All completions were excluded; see audit and use a suitable external verifier")
        # A hard link publishes atomically without replacing an existing file.
        os.link(partial_path, output_path)
        partial_path.unlink()
        audit["status"] = "complete"
        audit["output_sha256"] = _sha256(output_path)
    except BaseException as exc:
        audit["status"] = "failed" if isinstance(exc, Exception) else "interrupted"
        audit["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        audit["excluded_count"] = len(audit["excluded"])
        audit["parse_failure_count"] = len(audit["parse_failures"])
        with audit_path.open("w", encoding="utf-8") as handle:
            json.dump(audit, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return audit
