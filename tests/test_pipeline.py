import json
from pathlib import Path

import numpy as np
import pytest

from drift.cli import run_score, run_select
from drift.data import prepare_corpus
from drift.features import merge_features, load_feature_table
from drift.io import sha256_file, write_json, write_jsonl
from drift.masking import assistant_labels
from drift.projection import GrassProjector


class CharacterTokenizer:
    """Exact-offset tokenizer used to inspect which text the loss supervises."""
    all_special_tokens = ["$"]
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            truncation=False, max_length=None):
        text = "^"
        for i, row in enumerate(messages):
            text += "<" + row["role"] + ">" + row["content"]
            if row["role"] == "assistant":
                text += "$"
                if i < len(messages)-1:
                    text += "\n"
            else:
                text += "\n"
        if add_generation_prompt:
            text += "<assistant>"
        if tokenize:
            return self(text, truncation=truncation, max_length=max_length)["input_ids"]
        return text

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None,
                 return_offsets_mapping=False):
        if truncation:
            text = text[:max_length]
        result = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i+1) for i in range(len(text))]
        return result


def labeled_text(value):
    return "".join(chr(x) for x in value["labels"] if x != -100)


def test_multiturn_anchor_supervises_only_sampled_response():
    messages = [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "HISTORY"},
                {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "FINAL"}]
    tok = CharacterTokenizer()
    corpus = assistant_labels(tok, messages)
    anchor = assistant_labels(tok, messages, last_only=True, require_complete=True)
    assert labeled_text(corpus) == "HISTORY$FINAL$"
    assert labeled_text(anchor) == "FINAL$"
    assert corpus["input_ids"] == anchor["input_ids"]
    with pytest.raises(ValueError, match="truncated signed gradient"):
        assistant_labels(tok, messages, max_length=35, last_only=True, require_complete=True)


def test_assistant_only_and_zero_supervision_after_truncation():
    tok = CharacterTokenizer()
    assert labeled_text(assistant_labels(tok, [{"role": "assistant", "content": "Hello"}])) == "Hello$"
    rows = [{"role": "user", "content": "long "*100}, {"role": "assistant", "content": "answer"}]
    assert assistant_labels(tok, rows, max_length=30)["supervised_tokens"] == 0


def test_context_dependent_assistant_terminators():
    class ContextTokenizer(CharacterTokenizer):
        all_special_tokens = ["$", "~"]

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                                truncation=False, max_length=None):
            text = super().apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
            text = text.replace("$\n", "~\n")
            if tokenize:
                return self(text, truncation=truncation, max_length=max_length)["input_ids"]
            return text

    rows = [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}]
    tok = ContextTokenizer()
    assert labeled_text(assistant_labels(tok, rows)) == "A1~A2$"
    assert labeled_text(assistant_labels(tok, rows, last_only=True, require_complete=True)) == "A2$"


def test_context_mask_excludes_all_headers_and_uses_added_special_terminators():
    class SpecialToken:
        special = True

        def __str__(self):
            return "<|im_end|>"

    class AtomicTokenizer:
        # Some tokenizer configurations expose turn endings only through the
        # added-token decoder, not all_special_tokens.
        all_special_tokens = ["<|endoftext|>", "<|im_start|>"]
        added_tokens_decoder = {1002: SpecialToken()}
        specials = {"<|endoftext|>": 1000, "<|im_start|>": 1001, "<|im_end|>": 1002}

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                                truncation=False, max_length=None):
            text = "<|endoftext|>"
            for index, row in enumerate(messages):
                text += "<|im_start|>" + row["role"] + "\n" + row["content"]
                final_assistant = row["role"] == "assistant" and index == len(messages)-1
                text += "<|endoftext|>" if final_assistant else "<|im_end|>\n"
            if add_generation_prompt:
                text += "<|im_start|>assistant\n"
            return self(text, truncation=truncation, max_length=max_length)["input_ids"] if tokenize else text

        def __call__(self, text, add_special_tokens=False, truncation=False,
                     max_length=None, return_offsets_mapping=False):
            ids, offsets = [], []
            position = 0
            while position < len(text):
                special = next((token for token in self.specials if text.startswith(token, position)), None)
                end = position + (len(special) if special else 1)
                ids.append(self.specials[special] if special else ord(text[position]))
                offsets.append((position, end))
                position = end
            if truncation:
                ids, offsets = ids[:max_length], offsets[:max_length]
            result = {"input_ids": ids}
            if return_offsets_mapping:
                result["offset_mapping"] = offsets
            return result

    rows = [{"role": "system", "content": "SYSTEM_ONLY"},
            {"role": "user", "content": "USER_ONE"},
            {"role": "assistant", "content": "FIRST"},
            {"role": "user", "content": "USER_TWO"},
            {"role": "assistant", "content": "LAST"}]
    tok = AtomicTokenizer()
    corpus = assistant_labels(tok, rows)
    anchor = assistant_labels(tok, rows, last_only=True, require_complete=True)
    assert [value for value in corpus["labels"] if value != -100] == [
        *map(ord, "FIRST"), 1002, *map(ord, "LAST"), 1000]
    assert [value for value in anchor["labels"] if value != -100] == [*map(ord, "LAST"), 1000]
    assert all(label == -100 for token, label in zip(corpus["input_ids"], corpus["labels"]) if token == 1001)
    assert corpus["labels"][0] == -100  # Initial BOS equals EOS but is not a target.
    assert corpus["input_ids"] == anchor["input_ids"]

    class DifferentGenerationPrefix(AtomicTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            value = super().apply_chat_template(messages, **kwargs)
            if kwargs.get("add_generation_prompt"):
                return value + "different prefix"
            return value

    with pytest.raises(ValueError, match="generation prompt"):
        assistant_labels(DifferentGenerationPrefix(), rows, last_only=True, require_complete=True)


def test_grass_linearity_and_saved_coordinate_identity(tmp_path):
    layout = [("a", (5,)), ("b", (7,))]
    projector = GrassProjector(layout, dimension=3, seed=7, chunk_elements=6, multiplier=1)
    clone = GrassProjector(layout, dimension=3, seed=7, chunk_elements=6, multiplier=1)
    assert projector.fingerprint == clone.fingerprint
    x, y = np.arange(12, dtype=np.float32), np.arange(12, dtype=np.float32)[::-1]
    np.testing.assert_allclose(projector.project_numpy(x+2*y),
                               projector.project_numpy(x)+2*projector.project_numpy(y))
    assert len(projector.indices) == 6  # 3 sampled coordinates in each of two chunks.
    assert len(set(projector.indices.tolist())) == 6
    projector.save(tmp_path)
    with np.load(tmp_path/"projection.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["indices"], projector.indices)


def make_shards(root, rows, matrix, world=2, fingerprint="same-checkpoint", input_sha="corpus"):
    roots = []
    for rank in range(world):
        out = root/f"rank-{rank}";out.mkdir(parents=True)
        (out/"chunks").mkdir()
        indices = list(range(rank, len(rows), world))
        block = matrix[indices]
        metadata = [{**rows[i], "index": i} for i in indices]
        np.savez(out/"chunks/000.npz", features=block, indices=np.array(indices, dtype=np.int64))
        write_jsonl(out/"chunks/000.jsonl", metadata)
        write_jsonl(out/"rejected.jsonl", [])
        manifest = {"state": "complete", "rank": rank, "world_size": world,
                    "input_sha256": input_sha, "input_rows": len(rows),
                    "feature_rows": len(indices), "rejected_rows": 0,
                    "dimension": matrix.shape[1], "feature_fingerprint": fingerprint,
                    "identity": {}, "skip_no_supervision": False,
                    "rejected_sha256": sha256_file(out/"rejected.jsonl"),
                    "chunks": [{"array": "chunks/000.npz", "metadata": "chunks/000.jsonl",
                                "rows": len(indices), "array_sha256": sha256_file(out/"chunks/000.npz"),
                                "metadata_sha256": sha256_file(out/"chunks/000.jsonl")}]}
        write_json(out/"manifest.json", manifest);roots.append(out)
    return roots


def test_shard_merge_score_select_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    corpus_rows = [{"id": str(i), "messages": [{"role": "user", "content": f"question{i}"},
                    {"role": "assistant", "content": f"answer{i}"}]} for i in range(20)]
    corpus = tmp_path/"corpus.jsonl";write_jsonl(corpus, corpus_rows)
    train = rng.normal(size=(20, 4)).astype(np.float32)*np.arange(1, 21)[:, None]
    roots = make_shards(tmp_path/"train", corpus_rows, train, world=3, input_sha=sha256_file(corpus))
    merge_features(roots, tmp_path/"merged-train")
    loaded, records, _ = load_feature_table(tmp_path/"merged-train")
    np.testing.assert_array_equal(loaded, train.astype(np.float32))
    assert [r["id"] for r in records] == [str(i) for i in range(20)]
    assert not (tmp_path/"merged-train/unsorted.npy").exists()
    anchors = [{"id": f"v{i}", "task": f"t{i//2}", "query_id": f"q{i//2}",
                "reward": 1 if i % 2 == 0 else -1} for i in range(4)]
    roots = make_shards(tmp_path/"anchor", anchors, rng.normal(size=(4, 4)).astype(np.float32), input_sha="anchor")
    merge_features(roots, tmp_path/"merged-anchor")
    run_score(tmp_path/"merged-train", tmp_path/"merged-anchor", tmp_path/"scores")
    result = run_select(tmp_path/"scores", tmp_path/"selected", fraction=.2, corpus=corpus)
    assert result["selected"] == 4
    exported = [json.loads(x) for x in (tmp_path/"selected/subset.jsonl").read_text().splitlines()]
    assert len(exported) == 4
    assert all(row in corpus_rows for row in exported)


def test_merge_rejects_missing_rank_and_modified_chunk(tmp_path):
    roots = make_shards(tmp_path/"input", [{"id": str(i)} for i in range(4)], np.ones((4, 3)))
    with pytest.raises(ValueError, match="Missing or duplicated ranks"):
        merge_features(roots[:1], tmp_path/"missing")
    with (roots[1]/"chunks/000.jsonl").open("a") as f:
        f.write("{}\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        merge_features(roots, tmp_path/"corrupt")


def test_prepare_preserves_multiturn_and_rejects_duplicate_ids(tmp_path):
    messages = [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"},
                {"role": "user", "content": "C"}, {"role": "assistant", "content": "D"}]
    source = tmp_path/"source.jsonl";write_jsonl(source, [{"id": "a", "messages": messages}])
    output = tmp_path/"normalized.jsonl";prepare_corpus(source, output)
    assert json.loads(output.read_text())["messages"] == messages
    duplicate = tmp_path/"duplicate.jsonl";write_jsonl(duplicate, [{"id": "a", "messages": messages}]*2)
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_corpus(duplicate, tmp_path/"rejected.jsonl")
