"""Small CPU-only integrity checks for signed validation anchors."""

import json
import sys
import types
from pathlib import Path

import pytest

from drift.rewards import label_rollouts, normalize_exact, parse_choice
from drift.rollout import _request_seed, generate_rollouts


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def query(query_id="q", kind="choice", answer="K", task="validation"):
    return {
        "id": query_id, "task": task,
        "messages": [{"role": "system", "content": "Be precise."},
                     {"role": "user", "content": "Earlier question?"},
                     {"role": "assistant", "content": "Earlier answer."},
                     {"role": "user", "content": "Current question?"}],
        "verifier": {"type": kind, "answer": answer},
    }


def rollout(q, text="K", index=0, **extra):
    return {"id": f"{q['id']}:{index}", "query_id": q["id"], "task": q["task"],
            "messages": q["messages"] + [{"role": "assistant", "content": text}],
            "completion": text, "finish_reason": "stop", "generated_tokens": 1, **extra}


def inputs(tmp_path, queries, rollouts):
    return (write_jsonl(tmp_path / "queries.jsonl", queries),
            [write_jsonl(tmp_path / "rollouts.jsonl", rollouts)], tmp_path / "anchors.jsonl")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


@pytest.mark.parametrize("text,expected", [("K", "K"), ("(J)", "J"),
    ("The final answer is K.", "K"), (r"Therefore \boxed{L}.", "L"),
    ("A and K are plausible.", None), ("Answer: A. Answer: K.", None),
    ("Answer: A or K", "A"), ("J. N/A", "J"), ("H. 100", "H"),
    ("H. 100\nI. 200", None), ("", None)])
def test_choice_prediction_only(text, expected):
    assert parse_choice(text) == expected


def test_choice_extended_labels_and_choice_bounds():
    assert parse_choice("AA", [str(i) for i in range(27)]) == "AA"
    assert parse_choice("K", [str(i) for i in range(10)]) is None
    assert parse_choice("AA") is None


def test_exact_normalization_is_explicit():
    assert normalize_exact("  Ａ\n B  ") == "A B"
    assert normalize_exact("Answer") != normalize_exact("answer")
    assert normalize_exact("Answer", {"casefold": True}) == "answer"
    with pytest.raises(ValueError, match="Unknown"):
        normalize_exact("a", {"ignore_all_punctuation": True})


def test_positive_negative_and_complete_multiturn_are_preserved(tmp_path):
    q = query()
    args = inputs(tmp_path, [q], [rollout(q, "K"), rollout(q, "A", 1)])
    audit = label_rollouts(*args)
    anchors = read_jsonl(args[2])
    assert [row["reward"] for row in anchors] == [1, -1]
    assert anchors[0]["messages"] == q["messages"] + [{"role": "assistant", "content": "K"}]
    assert set(anchors[0]) == {"id", "query_id", "task", "reward", "messages"}
    assert audit["status"] == "complete"
    assert audit["reference_answers_injected"] is False
    assert audit["output_anchors"] == 2
    assert not Path(str(args[2]) + ".partial").exists()


def test_all_wrong_and_all_correct_queries_are_kept(tmp_path):
    first, second = query("wrong"), query("right")
    args = inputs(tmp_path, [first, second], [rollout(first, "B"), rollout(second)])
    audit = label_rollouts(*args)
    assert audit["query_anchor_counts"] == {"wrong": 1, "right": 1}
    assert [row["reward"] for row in read_jsonl(args[2])] == [-1, 1]


@pytest.mark.parametrize("change,error", [
    ({"task": "other"}, "Task mismatch"),
    ({"query_id": "unknown"}, "Unknown rollout"),
    ({"id": "q:00"}, "Rollout id"),
    ({"sample_index": 10}, "sample_index"),
    ({"messages": [{"role": "assistant", "content": "K"}]}, "messages"),
    ({"finish_reason": None}, "finish_reason"),
])
def test_reject_misalignment(tmp_path, change, error):
    q = query()
    args = inputs(tmp_path, [q], [rollout(q, **change)])
    with pytest.raises(ValueError, match=error):
        label_rollouts(*args)
    assert not args[2].exists()


def test_duplicate_rollouts_and_missing_query_coverage(tmp_path):
    q = query()
    args = inputs(tmp_path, [q], [rollout(q), rollout(q)])
    with pytest.raises(ValueError, match="Duplicate"):
        label_rollouts(*args)
    write_jsonl(args[0], [q, query("missing")])
    write_jsonl(args[1][0], [rollout(q)])
    with pytest.raises(ValueError, match="Missing rollout coverage"):
        label_rollouts(*args)


@pytest.mark.parametrize("text,extra", [("", {}), ("K", {"finish_reason": "length"}),
                                       ("K", {"truncated": True}), ("K", {"finish_reason": "abort"})])
def test_bad_generation_default_refuses_publish(tmp_path, text, extra):
    q = query()
    args = inputs(tmp_path, [q], [rollout(q), rollout(q, text, 1, **extra)])
    with pytest.raises(ValueError, match="allow_exclusions"):
        label_rollouts(*args)
    assert not args[2].exists()
    audit = json.loads(Path(str(args[2]) + ".audit.json").read_text())
    assert audit["status"] == "failed"
    assert audit["excluded"][0]["id"] == "q:1"
    assert audit["negative"] == 0


def test_explicit_exclusions_report_disappearing_queries(tmp_path):
    good, bad = query("good"), query("bad")
    args = inputs(tmp_path, [good, bad], [rollout(good), rollout(bad, "")])
    audit = label_rollouts(*args, allow_exclusions=True)
    assert audit["zero_anchor_query_ids"] == ["bad"]
    assert audit["excluded_count"] == 1
    assert len(read_jsonl(args[2])) == 1


def test_no_all_excluded_output(tmp_path):
    q = query()
    args = inputs(tmp_path, [q], [rollout(q, "")])
    with pytest.raises(ValueError, match="All completions"):
        label_rollouts(*args, allow_exclusions=True)
    assert not args[2].exists()


def test_external_labels_are_strict_and_never_execute_code(tmp_path):
    q = query(kind="external")
    marker = tmp_path / "must_not_exist"
    generated_code = f"open({str(marker)!r}, 'w').write('executed')"
    args = inputs(tmp_path, [q], [rollout(q, generated_code)])
    external = write_jsonl(tmp_path / "official.jsonl", [{"id": "q:0", "reward": 1}])
    audit = label_rollouts(*args, external_labels=external)
    assert audit["positive"] == 1
    assert not marker.exists()


@pytest.mark.parametrize("labels,error", [([], "exactly cover"),
    ([{"id": "q:0", "reward": 0}], "exactly -1 or"),
    ([{"id": "q:0", "reward": True}], "exactly -1 or"),
    ([{"id": "q:0", "reward": 1}, {"id": "other:0", "reward": -1}], "exactly cover"),
    ([{"id": "q:0", "reward": 1}, {"id": "q:0", "reward": -1}], "duplicate"),
])
def test_external_integrity(tmp_path, labels, error):
    q = query(kind="external")
    args = inputs(tmp_path, [q], [rollout(q, "code")])
    external = write_jsonl(tmp_path / "official.jsonl", labels)
    with pytest.raises(ValueError, match=error):
        label_rollouts(*args, external_labels=external)


def test_math_normal_parse_failure_is_negative_and_retained(tmp_path, monkeypatch):
    q = query(kind="math", answer="1")
    args = inputs(tmp_path, [q], [rollout(q, "unparseable"), rollout(q, "1", 1)])
    monkeypatch.setattr("drift.rewards._load_math_verifier", lambda: (
        lambda value, **kwargs: [] if value == "unparseable" else [value],
        lambda gold, pred: gold == pred))
    audit = label_rollouts(*args)
    assert audit["positive"] == 1 and audit["negative"] == 1
    assert audit["parse_failures"][0]["reason"] == "math_no_parsed_answer"
    assert audit["excluded_count"] == 0
    assert len(read_jsonl(args[2])) == 2


def test_math_verifier_exception_cannot_turn_into_a_negative(tmp_path, monkeypatch):
    q = query(kind="math", answer="1")
    args = inputs(tmp_path, [q], [rollout(q, "1")])

    def broken(*_):
        raise TimeoutError("verification timed out")

    monkeypatch.setattr("drift.rewards._load_math_verifier", lambda: (lambda value, **_: [value], broken))
    with pytest.raises(ValueError, match="allow_exclusions"):
        label_rollouts(*args)
    audit = json.loads(Path(str(args[2]) + ".audit.json").read_text())
    assert audit["negative"] == 0
    assert audit["excluded"][0]["reason"] == "math_unverifiable_TimeoutError"


def test_invalid_math_reference_fails_before_publishing(tmp_path, monkeypatch):
    q = query(kind="math", answer="1")
    args = inputs(tmp_path, [q], [rollout(q, "1")])
    monkeypatch.setattr("drift.rewards._load_math_verifier", lambda: (lambda *_, **__: [], lambda *_: True))
    with pytest.raises(ValueError, match="reference is unverifiable"):
        label_rollouts(*args)
    assert not args[2].exists()


def test_refuses_overwrite(tmp_path):
    q = query()
    args = inputs(tmp_path, [q], [rollout(q)])
    label_rollouts(*args)
    original = args[2].read_bytes()
    with pytest.raises(FileExistsError):
        label_rollouts(*args)
    assert args[2].read_bytes() == original


def test_generation_parameter_checks_do_not_import_gpu_libraries(tmp_path):
    with pytest.raises(ValueError, match="rank"):
        generate_rollouts("unused", tmp_path / "unused", "unused", rank=2, world_size=2)
    with pytest.raises(ValueError, match="top_k"):
        generate_rollouts("unused", tmp_path / "unused", "unused", top_k=0)


def test_request_seeds_do_not_depend_on_rank_or_python_hash():
    assert _request_seed(42, "q", 0) == _request_seed(42, "q", 0)
    assert _request_seed(42, "q", 0) != _request_seed(42, "q", 1)
    assert _request_seed(42, "q", 0) != _request_seed(43, "q", 0)


def test_exact_regex_is_fixed_and_preserves_failures(tmp_path):
    q = query(kind="exact", answer="42")
    q["verifier"]["extract_regex"] = r"ANSWER:\s*(\d+)"
    args = inputs(tmp_path, [q], [rollout(q, "Reasoning. ANSWER: 42"),
                                rollout(q, "Reasoning without marker", 1),
                                rollout(q, "ANSWER: 7 ANSWER: 42", 2)])
    audit = label_rollouts(*args)
    assert [row["reward"] for row in read_jsonl(args[2])] == [1, -1, -1]
    assert audit["parse_failure_count"] == 2


@pytest.mark.parametrize("pattern", ["ANSWER", "(A)(B)", "("])
def test_exact_regex_requires_one_capture(tmp_path, pattern):
    q = query(kind="exact", answer="42")
    q["verifier"]["extract_regex"] = pattern
    args = inputs(tmp_path, [q], [rollout(q, "42")])
    with pytest.raises(ValueError, match="extract_regex"):
        label_rollouts(*args)


def test_tool_messages_are_rejected(tmp_path):
    q = query()
    q["messages"].insert(1, {"role": "tool", "content": "unsupported"})
    args = inputs(tmp_path, [q], [rollout(q)])
    with pytest.raises(ValueError, match="invalid message"):
        label_rollouts(*args)


@pytest.mark.parametrize("rows", [(0, 2), (1,)])
def test_missing_sample_indices_are_rejected(tmp_path, rows):
    q = query()
    args = inputs(tmp_path, [q], [rollout(q, index=index) for index in rows])
    with pytest.raises(ValueError, match="contiguous"):
        label_rollouts(*args)


def test_mismatched_sample_counts_are_rejected(tmp_path):
    first, second = query("first"), query("second")
    args = inputs(tmp_path, [first, second], [rollout(first), rollout(first, index=1), rollout(second)])
    with pytest.raises(ValueError, match="Unequal"):
        label_rollouts(*args)


def test_external_metadata_cannot_override_query_task(tmp_path):
    q = query(kind="external")
    args = inputs(tmp_path, [q], [rollout(q)])
    labels = write_jsonl(tmp_path / "official.jsonl", [{"id": "q:0", "reward": 1, "task": "wrong"}])
    with pytest.raises(ValueError, match="task mismatch"):
        label_rollouts(*args, external_labels=labels)


def test_vllm_rollout_contract_without_loading_a_model(tmp_path, monkeypatch):
    calls = {}

    class FakeTokenizer:
        name_or_path = "fake-model"
        init_kwargs = {"_commit_hash": "tokenizer-commit"}
        model_max_length = 256
        eos_token_id = 999

        def get_chat_template(self):
            return "complete conversation template"

        def save_pretrained(self, path):
            path.mkdir()
            (path / "tokenizer_config.json").write_text('{"fake":true}')

        def apply_chat_template(self, messages, **kwargs):
            calls.setdefault("messages", []).append(messages)
            assert kwargs == {"tokenize": True, "add_generation_prompt": True}
            return [10, 11, 12]

    class FakeConfig:
        _commit_hash = "model-commit"
        max_position_embeddings = 256
        eos_token_id = 2

        def save_pretrained(self, path):
            path.mkdir()
            (path / "config.json").write_text('{"fake":true}')

        def to_dict(self):
            return {"eos_token_id": 2}

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *_, **__: FakeTokenizer())
    transformers.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *_, **__: FakeConfig())
    transformers.GenerationConfig = types.SimpleNamespace(from_pretrained=lambda *_, **__: types.SimpleNamespace(eos_token_id=[2, 9]))
    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = lambda **kwargs: kwargs

    class FakeEngine:
        def __init__(self, **kwargs):
            calls["engine"] = kwargs

        def generate(self, prompts, sampling_params, **kwargs):
            calls["parameters"] = sampling_params
            assert all(prompt == {"prompt_token_ids": [10, 11, 12]} for prompt in prompts)
            outputs = []
            for index, _ in enumerate(prompts):
                text, tokens, reason = [("K", [21, 9], "stop"), ("", [9], "stop"), ("unfinished", [21, 22], "length")][index % 3]
                completion = types.SimpleNamespace(text=text, token_ids=tokens, finish_reason=reason, stop_reason=9 if reason == "stop" else None)
                outputs.append(types.SimpleNamespace(outputs=[completion]))
            return outputs

    vllm.LLM = FakeEngine
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    queries = [query("first"), query("second"), query("third")]
    source = write_jsonl(tmp_path / "queries.jsonl", queries)
    output_dir = tmp_path / "rank0"
    manifest = generate_rollouts(source, output_dir, "fake-model", samples=3, rank=0, world_size=2, max_new_tokens=8)
    rows = read_jsonl(output_dir / "rollouts.jsonl")
    assert manifest["status"] == "complete"
    assert manifest["assigned_query_ids"] == ["first", "third"]
    assert len(rows) == 6
    assert manifest["empty_completions"] == 2
    assert manifest["truncated_completions"] == 2
    assert calls["messages"] == [queries[0]["messages"], queries[2]["messages"]]
    assert all(parameter["stop_token_ids"] == [2, 9] for parameter in calls["parameters"])
    assert rows[0]["messages"][:-1] == queries[0]["messages"]
    assert rows[0]["completion"] == "K"
    assert rows[1]["empty"] is True
    assert rows[2]["truncated"] is True
    assert manifest["resolved_model_revision"] == "model-commit"
    assert manifest["tokenizer"]["resolved_revision"] == "tokenizer-commit"
    assert manifest["tokenizer"]["files"]
    with pytest.raises(FileExistsError):
        generate_rollouts(source, output_dir, "fake-model", max_new_tokens=8)
