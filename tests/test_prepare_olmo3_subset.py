"""Frozen subset reconstruction without network access, models, or large data."""
import gzip
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from drift.io import canonical_json, object_hash, read_jsonl, sha256_file


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_olmo3_subset.py"
spec = importlib.util.spec_from_file_location("prepare_olmo3_subset", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def conversation(text):
    return [{"role": "system", "content": "Keep every turn."},
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": "第二问"},
            {"role": "assistant", "content": text}]


def selected(index, messages, identifier="repeated-original-id"):
    return {"index": index, "id": identifier, "messages_sha256": object_hash(messages)}


def setup(tmp_path, selection, corpus):
    ledger = tmp_path / "selection.jsonl.gz"
    with gzip.open(ledger, "wt", encoding="utf-8") as stream:
        for row in selection:
            stream.write(canonical_json(row) + "\n")
    manifest = tmp_path / "selection_manifest.json"
    manifest.write_text(json.dumps({"corpus": {"repo": "fixture/corpus", "revision": "fixed-commit", "split": "train"},
                                    "selection": {"count": len(selection), "sha256": sha256_file(ledger)}}))
    source = tmp_path / "corpus.jsonl"
    source.write_text("".join(canonical_json(row) + "\n" for row in corpus), encoding="utf-8")
    return module.parser().parse_args(["--selection", str(ledger), "--manifest", str(manifest),
                                      "--corpus", str(source), "--output", str(tmp_path / "subset.jsonl")])


def test_rebuild_preserves_multiturn_unicode_duplicates_and_selected_order(tmp_path):
    first, second = conversation("中文答案"), conversation("another answer")
    nullable = [dict(row, function_calls=None, functions=[]) for row in first]
    # All original IDs collide; source order and selected order also differ.
    args = setup(tmp_path, [selected(7, first), selected(1, second), selected(2, first)],
                 [{"id": "same", "messages": nullable}, {"id": "same", "messages": first},
                  {"id": "same", "messages": conversation("not selected")}, {"id": "same", "messages": second}])
    report = module.rebuild(args)
    rows = list(read_jsonl(args.output))
    assert [row["id"] for row in rows] == ["drift-paper:1", "drift-paper:2", "drift-paper:7"]
    assert [row["messages"] for row in rows] == [second, first, first]
    assert report["rows"] == 3 and report["unique_selected_messages"] == 2
    assert report["corpus_rows_scanned"] == 4
    assert report["sha256"] == sha256_file(args.output)
    assert json.loads(Path(args.output + ".manifest.json").read_text())["state"] == "complete"
    assert not list(tmp_path.glob(".drift-*"))


def test_missing_hashes_fail_without_partial_output_and_keep_multiplicity(tmp_path):
    first, absent = conversation("present"), conversation("absent")
    args = setup(tmp_path, [selected(9, absent), selected(1, first), selected(5, absent)],
                 [{"messages": first}])
    with pytest.raises(module.MissingConversations, match="2 selected rows"):
        module.rebuild(args)
    assert not Path(args.output).exists()
    missing = list(read_jsonl(args.output + ".missing.jsonl"))
    assert [row["index"] for row in missing] == [5, 9]
    report = json.loads(Path(args.output + ".manifest.json").read_text())
    assert report["state"] == "missing" and report["missing_count"] == 2


@pytest.mark.parametrize("mutation", ["digest", "count", "duplicate_index"])
def test_invalid_selection_fails_before_reading_corpus(tmp_path, monkeypatch, mutation):
    messages = conversation("answer")
    ledger = [selected(0, messages)]
    if mutation == "duplicate_index":
        ledger.append(selected(0, messages))
    args = setup(tmp_path, ledger, [])
    manifest = json.loads(Path(args.manifest).read_text())
    if mutation == "digest":
        manifest["selection"]["sha256"] = "0" * 64
    elif mutation == "count":
        manifest["selection"]["count"] += 1
    Path(args.manifest).write_text(json.dumps(manifest))
    reader = Mock(side_effect=AssertionError("Corpus should not be read"))
    monkeypatch.setattr(module, "corpus_rows", reader)
    with pytest.raises(ValueError):
        module.rebuild(args)
    reader.assert_not_called()
    assert not Path(args.output).exists()


def test_streaming_requires_revision_and_passes_it_to_loader(tmp_path, monkeypatch):
    messages = conversation("answer")
    args = setup(tmp_path, [selected(3, messages)], [])
    args.corpus = None
    manifest = json.loads(Path(args.manifest).read_text())
    manifest["corpus"].pop("revision")
    Path(args.manifest).write_text(json.dumps(manifest))
    loader = Mock(return_value=iter([{"messages": messages}]))
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=loader))
    with pytest.raises(ValueError, match="explicit revision"):
        module.rebuild(args)
    loader.assert_not_called()
    args.revision = "user-pinned-revision"
    module.rebuild(args)
    loader.assert_called_once_with("fixture/corpus", split="train", revision="user-pinned-revision", streaming=True)


def test_existing_output_is_never_overwritten(tmp_path):
    messages = conversation("answer")
    args = setup(tmp_path, [selected(0, messages)], [{"messages": messages}])
    Path(args.output).write_text("keep original\n")
    with pytest.raises(FileExistsError):
        module.rebuild(args)
    assert Path(args.output).read_text() == "keep original\n"


def test_declared_output_digest_checked_before_publication(tmp_path):
    messages = conversation("answer")
    args = setup(tmp_path, [selected(0, messages)], [{"messages": messages}])
    manifest = json.loads(Path(args.manifest).read_text())
    manifest["output"] = {"sha256": "0" * 64}
    Path(args.manifest).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Output sha256"):
        module.rebuild(args)
    assert not Path(args.output).exists()


def test_nonempty_extra_fields_cannot_silently_drop_tool_content(tmp_path):
    messages = conversation("answer")
    messages[-1]["function_calls"] = [{"name": "real_tool", "arguments": "{}"}]
    args = setup(tmp_path, [selected(0, conversation("answer"))], [{"messages": messages}])
    with pytest.raises(ValueError, match="nonempty message field"):
        module.rebuild(args)
    assert not Path(args.output).exists()


def test_explicit_source_filter_excludes_null_source_tools_before_validation(tmp_path):
    tool_messages = conversation("tool response")
    tool_messages[-1]["function_calls"] = [{"name": "real_tool", "arguments": "{}"}]
    messages = conversation("selected answer")
    args = setup(tmp_path, [selected(0, messages)],
                 [{"source": None, "messages": tool_messages},
                  {"source": "ordinary-source", "messages": messages}])
    # The tool-bearing row is still an error unless this filter is explicit.
    with pytest.raises(ValueError, match="nonempty message field"):
        module.rebuild(args)
    manifest = json.loads(Path(args.manifest).read_text())
    manifest["corpus"]["filter"] = {"source_not_null": True}
    Path(args.manifest).write_text(json.dumps(manifest))
    report = module.rebuild(args)
    assert list(read_jsonl(args.output)) == [{"id": "drift-paper:0", "messages": messages}]
    assert report["source"]["filter"] == {"source_not_null": True}
    assert report["corpus_rows_scanned"] == 2
    assert report["corpus_rows_skipped_by_filter"] == 1
