# Released Olmo3 selection

`olmo3_selected.jsonl.gz` lists the 192,453 conversations DRIFT selected for the paper's Olmo3 run, in the order they were used for training. It holds identifiers and hashes only. The conversations are rebuilt from the public Dolci corpus:

```bash
python scripts/prepare_olmo3_subset.py --output data/olmo3_drift.jsonl
```

## Index format

One JSON object per line:

```json
{"index":0,"id":"allenai/correct-python-sft-187k-decontam-v2_tmp_ids_170766","messages_sha256":"8b5e44b5d393aa273821cc526526db8d2fcb4ad60334c588574b7f44feb25300"}
```

| Field | Meaning |
|---|---|
| `index` | Position in the training order |
| `id` | Source ID in the corpus. IDs are not unique there, so rows are matched by hash |
| `messages_sha256` | SHA-256 of the conversation's `role` and `content` turns, serialized as compact UTF-8 JSON with sorted keys |

In Python the hash is `sha256(json.dumps(turns, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())`, where `turns` is the list of `{"role", "content"}` objects.

The corpus contains repeated conversations, and the selection keeps them: the 192,453 rows cover 179,802 distinct conversations.

## Manifest

[`olmo3_selection_manifest.json`](olmo3_selection_manifest.json) pins what the index refers to.

| Item | Value |
|---|---|
| Model | `allenai/Olmo-3-7B-Instruct-SFT` at `d9687e20d263cd0876dea143723b3c4974d19935` |
| Corpus | `allenai/Dolci-Instruct-SFT` at `b779ae7ae5c757ccea6abb04a05204470bf1e602`, train split |
| Candidates | About 2M rows, those with a non-null `source` |
| Selected | 192,453 rows |

It also stores the SHA-256 of the index file and a digest of the ordered message hashes. The rebuild script checks both.
