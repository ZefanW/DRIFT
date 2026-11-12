"""Convert original conversation records without flattening multi-turn context."""
from __future__ import annotations

import json
from pathlib import Path

from .io import canonical_json, read_jsonl, sha256_file, validate_messages, write_json


def iter_source(path):
    path = Path(path)
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if not files:
            raise ValueError("Directory contains no Parquet shards")
        for file in files:
            yield from iter_source(file)
    elif path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError("Parquet input requires pip install 'drift-attribution[data]'") from error
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    elif path.suffix in {".jsonl", ".ndjson"}:
        yield from read_jsonl(path)
    else:
        raise ValueError("Input must be JSONL, Parquet, or a directory of Parquet shards")


def prepare_corpus(source, output):
    output = Path(output)
    if output.exists() or output.with_suffix(output.suffix + ".manifest.json").exists():
        raise FileExistsError(output)
    seen = set()
    count = 0
    with output.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(iter_source(source)):
            messages = row.get("messages")
            validate_messages(messages)
            if not any(message["role"] == "assistant" for message in messages):
                raise ValueError(f"Corpus row {index} has no assistant response")
            identifier = str(row.get("id", f"row:{index}"))
            if not identifier or identifier in seen:
                raise ValueError(f"Duplicate or empty corpus ID at row {index}: {identifier!r}")
            seen.add(identifier)
            record = {"id": identifier, "messages": messages}
            if "source" in row:
                record["source"] = row["source"]
            stream.write(canonical_json(record) + "\n")
            count += 1
    if not count:
        raise ValueError("Empty corpus")
    manifest = {"schema_version": 1, "rows": count, "sha256": sha256_file(output),
                "ordering": "Input JSONL order or sorted Parquet filenames, preserving each shard's row order",
                "messages_preserved": True}
    write_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    return manifest


def export_subset(corpus, selected_ids, output):
    """Keep original corpus order and every message; reject missing/duplicate IDs."""
    selected = list(selected_ids)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Selected IDs must be nonempty and unique")
    pending = set(selected)
    seen = set()
    count = 0
    with open(output, "x", encoding="utf-8") as stream:
        for row in read_jsonl(corpus):
            identifier = row["id"]
            if identifier in seen:
                raise ValueError(f"Duplicate corpus ID: {identifier}")
            seen.add(identifier)
            if identifier in pending:
                stream.write(canonical_json(row) + "\n")
                pending.remove(identifier)
                count += 1
    if pending:
        raise ValueError(f"Selected IDs missing from corpus: {sorted(pending)[:5]}")
    return {"rows": count, "sha256": sha256_file(output)}
