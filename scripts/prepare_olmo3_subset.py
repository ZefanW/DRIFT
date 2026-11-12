#!/usr/bin/env python3
"""Rebuild the paper's frozen OLMo3 subset by exact full-message hashes.

Only the small selection ledger is distributed. Conversation contents are read
from a local corpus or a revision-pinned, streaming Hugging Face dataset.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile

from drift.data import iter_source
from drift.io import canonical_json, sha256_file, validate_messages


ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
SHA256 = re.compile(r"[0-9a-f]{64}")


class MissingConversations(ValueError):
    """The corpus did not contain every selected message hash."""


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", default=str(ARTIFACTS / "olmo3_selected.jsonl.gz"))
    p.add_argument("--manifest", default=str(ARTIFACTS / "olmo3_selection_manifest.json"))
    p.add_argument("--output", required=True, help="New canonical training JSONL")
    p.add_argument("--corpus", help="Local JSONL, Parquet, or directory of Parquet shards")
    p.add_argument("--repo", help="Override manifest corpus.repo for streaming input")
    p.add_argument("--revision", help="Explicit Hugging Face dataset revision; never inferred")
    p.add_argument("--split", help="Override manifest corpus.split (otherwise train)")
    p.add_argument("--cache-dir", help="Temporary disk cache parent; defaults to the output parent")
    return p


def read_spec(args):
    spec = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("Manifest must be an object")
    selection = spec.get("selection")
    if not isinstance(selection, dict) or type(selection.get("count")) is not int or selection["count"] < 1:
        raise ValueError("Manifest selection.count must be a positive integer")
    if not isinstance(selection.get("sha256"), str) or not SHA256.fullmatch(selection["sha256"]):
        raise ValueError("Manifest selection.sha256 must hash the selection file bytes")
    if sha256_file(args.selection) != selection["sha256"]:
        raise ValueError("Selection file SHA256 does not match its manifest")
    corpus = spec.get("corpus", {})
    if not isinstance(corpus, dict):
        raise ValueError("Manifest corpus must be an object")
    corpus_filter = corpus.get("filter", {})
    if not isinstance(corpus_filter, dict) or set(corpus_filter) - {"source_not_null"}:
        raise ValueError("Manifest corpus.filter supports only source_not_null")
    if "source_not_null" in corpus_filter and type(corpus_filter["source_not_null"]) is not bool:
        raise ValueError("Manifest corpus.filter.source_not_null must be boolean")
    source = {"repo": args.repo or corpus.get("repo"),
              "revision": args.revision or corpus.get("revision"),
              "split": args.split or corpus.get("split", "train"),
              "filter": corpus_filter}
    if args.corpus:
        source["local_path"] = str(Path(args.corpus).resolve())
    else:
        for key in ("repo", "revision", "split"):
            if not isinstance(source[key], str) or not source[key].strip():
                raise ValueError(f"Streaming corpus requires an explicit {key} in the manifest or CLI")
    return spec, source


def selection_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or type(row.get("index")) is not int or not 0 <= row["index"] <= 2**63 - 1:
                raise ValueError(f"Selection line {line_number} needs a nonnegative integer index")
            if not isinstance(row.get("id"), str):
                raise ValueError(f"Selection line {line_number} needs its original string id")
            digest = row.get("messages_sha256")
            if not isinstance(digest, str) or not SHA256.fullmatch(digest):
                raise ValueError(f"Invalid messages_sha256 on selection line {line_number}")
            yield row


def canonical_messages(messages):
    """Normalize nullable Parquet fields without dropping real tool content."""
    validate_messages(messages)
    result = []
    for message in messages:
        for key, value in message.items():
            if key not in {"role", "content"} and value is not None and value not in ("", [], {}):
                raise ValueError(f"Cannot discard nonempty message field {key!r}")
        result.append({"role": message["role"], "content": message["content"]})
    return result


def corpus_rows(args, source):
    if args.corpus:
        yield from iter_source(args.corpus)
    else:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise ImportError("Streaming input requires the data extra: pip install -e '.[data]'") from error
        yield from load_dataset(source["repo"], split=source["split"],
                                revision=source["revision"], streaming=True)


def publish_json(path, value, workspace):
    """Publish without overwriting, even if another process creates the path."""
    temporary = Path(workspace) / (Path(path).name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.link(temporary, path)


def rebuild(args):
    output = Path(args.output)
    report_path = Path(str(output) + ".manifest.json")
    missing_path = Path(str(output) + ".missing.jsonl")
    for path in (output, report_path, missing_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Output already exists: {path}")
    spec, source = read_spec(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_parent = Path(args.cache_dir) if args.cache_dir else output.parent
    cache_parent.mkdir(parents=True, exist_ok=True)
    # SQLite holds both the ledger and matched conversations. RAM usage does
    # not scale with the total text size of the selected subset.
    with tempfile.TemporaryDirectory(prefix=".drift-subset-", dir=cache_parent) as cache:
        connection = sqlite3.connect(Path(cache) / "matches.sqlite")
        try:
            connection.execute("PRAGMA cache_size = -16384")
            connection.execute("PRAGMA temp_store = FILE")
            connection.execute("CREATE TABLE chosen (idx INTEGER PRIMARY KEY, source_id TEXT NOT NULL, digest TEXT NOT NULL)")
            connection.execute("CREATE TABLE contents (digest TEXT PRIMARY KEY, messages TEXT)")
            count = 0
            for row in selection_rows(args.selection):
                try:
                    connection.execute("INSERT INTO chosen VALUES (?, ?, ?)",
                                       (row["index"], row["id"], row["messages_sha256"]))
                except sqlite3.IntegrityError as error:
                    raise ValueError(f"Duplicate selection index: {row['index']}") from error
                connection.execute("INSERT OR IGNORE INTO contents (digest) VALUES (?)", (row["messages_sha256"],))
                count += 1
            if count != spec["selection"]["count"]:
                raise ValueError(f"Selection count mismatch: expected {spec['selection']['count']}, found {count}")
            connection.commit()
            unique = connection.execute("SELECT COUNT(*) FROM contents").fetchone()[0]
            matched = scanned = skipped = 0
            for row in corpus_rows(args, source):
                scanned += 1
                if not isinstance(row, dict):
                    raise ValueError(f"Corpus row {scanned} must be an object")
                if source["filter"].get("source_not_null", False):
                    if "source" not in row:
                        raise ValueError(f"Corpus row {scanned} lacks source required by the declared filter")
                    if row["source"] is None:
                        skipped += 1
                        continue
                if "messages" not in row:
                    raise ValueError(f"Corpus row {scanned} has no messages field")
                messages = canonical_messages(row["messages"])
                text = canonical_json(messages)
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                existing = connection.execute("SELECT messages FROM contents WHERE digest = ?", (digest,)).fetchone()
                if existing is None:
                    continue
                if existing[0] is not None:
                    if existing[0] != text:
                        raise ValueError("Different messages share a selected SHA256 digest")
                    continue
                if not any(message["role"] == "assistant" for message in messages):
                    raise ValueError("A selected conversation has no assistant response")
                connection.execute("UPDATE contents SET messages = ? WHERE digest = ?", (text, digest))
                matched += 1
                if matched % 256 == 0:
                    connection.commit()
                if matched == unique:
                    break  # No corpus order or ID assumptions are needed.
            connection.commit()
            report = {"schema_version": 1, "source": source, "selection_count": count,
                      "selection_sha256": spec["selection"]["sha256"],
                      "selection_manifest_sha256": sha256_file(args.manifest),
                      "unique_selected_messages": unique, "matched_unique_messages": matched,
                      "corpus_rows_scanned": scanned,
                      "corpus_rows_skipped_by_filter": skipped,
                      "ordering": "ascending selection.index; corpus IDs and order are not used",
                      "messages_hash_definition": "SHA256 of UTF-8 json.dumps(role/content turns, ensure_ascii=False, sort_keys=True, separators=(',', ':')); only empty extra fields may be omitted"}
            if matched != unique:
                missing = 0
                # The sidecar has one record per missing selection, including
                # repeated selected occurrences of the same conversation.
                with missing_path.open("x", encoding="utf-8") as stream:
                    for index, identifier, digest in connection.execute(
                        "SELECT idx, source_id, chosen.digest FROM chosen JOIN contents USING (digest) "
                        "WHERE messages IS NULL ORDER BY idx"
                    ):
                        stream.write(canonical_json({"index": index, "id": identifier, "messages_sha256": digest}) + "\n")
                        missing += 1
                report.update(state="missing", missing_count=missing, missing_file=str(missing_path))
                # Use the output filesystem for atomic publication even when
                # the conversation cache lives on a different mount.
                with tempfile.TemporaryDirectory(prefix=".drift-report-", dir=output.parent) as staging:
                    publish_json(report_path, report, staging)
                raise MissingConversations(f"{missing} selected rows are missing; see {missing_path}")
            # Stage only the final output on its destination filesystem so the
            # final hard-link publication is exclusive and atomic.
            with tempfile.TemporaryDirectory(prefix=".drift-output-", dir=output.parent) as staging:
                temporary = Path(staging) / "subset.jsonl"
                file_digest, message_digest = hashlib.sha256(), hashlib.sha256()
                written = 0
                with temporary.open("xb") as stream:
                    for index, expected, text in connection.execute(
                        "SELECT idx, chosen.digest, messages FROM chosen JOIN contents USING (digest) ORDER BY idx"
                    ):
                        if hashlib.sha256(text.encode("utf-8")).hexdigest() != expected:
                            raise ValueError(f"Cached message checksum mismatch for selection {index}")
                        line = (canonical_json({"id": f"drift-paper:{index}", "messages": json.loads(text)}) + "\n").encode("utf-8")
                        stream.write(line)
                        file_digest.update(line)
                        message_digest.update((expected + "\n").encode("ascii"))
                        written += 1
                if written != count:
                    raise ValueError("Output count does not match the frozen selection")
                actual = file_digest.hexdigest()
                if sha256_file(temporary) != actual:
                    raise ValueError("Written output checksum mismatch")
                report.update(state="complete", rows=written, sha256=actual,
                              ordered_messages_sha256=message_digest.hexdigest(),
                              ordered_messages_hash_definition="SHA256 of each selected messages_sha256 plus LF, in ascending selection.index order")
                expected_output = spec.get("output", {})
                if not isinstance(expected_output, dict):
                    raise ValueError("Manifest output must be an object")
                for key in ("rows", "sha256", "ordered_messages_sha256"):
                    if key in expected_output and expected_output[key] != report[key]:
                        raise ValueError(f"Output {key} does not match the selection manifest")
                os.link(temporary, output)
                publish_json(report_path, report, staging)
                return report
        finally:
            connection.close()


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        report = rebuild(args)
        print(json.dumps({"output": args.output, "rows": report["rows"], "sha256": report["sha256"]}, indent=2))
        return 0
    except (OSError, ValueError, ImportError, sqlite3.Error) as error:
        print(f"prepare_olmo3_subset: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
