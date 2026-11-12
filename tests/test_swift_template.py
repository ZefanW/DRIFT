"""CPU-only checks of the Swift adapter's encoding and collation contract.

The stub follows Swift 3.12's outer conversion/length handling and list-based
packing/padding contract. It does not claim a real Swift training integration.
"""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from drift.masking import assistant_labels


class ChatTokenizer:
    is_fast = True
    chat_template = "context-dependent assistant terminators"
    truncation_side = "right"
    all_special_tokens = ["$", "~"]
    added_tokens_decoder = {}
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            truncation=False, max_length=None):
        text = "^"
        for index, row in enumerate(messages):
            text += "<" + row["role"] + ">" + row["content"]
            if row["role"] == "assistant":
                text += "$" if index == len(messages)-1 else "~\n"
            else:
                text += "\n"
        if add_generation_prompt:
            text += "<assistant>"
        return self(text, truncation=truncation, max_length=max_length)["input_ids"] if tokenize else text

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None,
                 return_offsets_mapping=False):
        if truncation:
            text = text[:max_length]
        result = {"input_ids": list(map(ord, text))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i+1) for i in range(len(text))]
        return result


class SwiftTemplateStub:
    def __init__(self, processor, template_meta, default_system=None, max_length=None, **kwargs):
        self.tokenizer = processor
        self.template_meta = template_meta
        self.max_length = max_length
        self.truncation_strategy = kwargs.get("truncation_strategy", "raise")
        self.mode = "pt"
        self.task_type = "causal_lm"

    def encode(self, inputs, return_template_inputs=False, return_length=False):
        if isinstance(inputs, dict):
            messages = deepcopy(inputs["messages"])
            system = messages.pop(0)["content"] if messages[0]["role"] == "system" else None
            chosen = SimpleNamespace(messages=messages, system=system)
        else:
            chosen = inputs.chosen
        encoded = self._encode_truncated(chosen)
        encoded = {key: value for key, value in encoded.items() if value is not None}
        if not return_length:
            encoded.pop("length", None)
        if return_template_inputs:
            encoded["template_inputs"] = chosen
        return encoded

    def packing_row(self, rows):
        # Same list concatenation / position reset contract as Swift's base.
        return {"input_ids": sum((row["input_ids"] for row in rows), []),
                "labels": sum((row["labels"] for row in rows), []),
                "position_ids": sum((list(range(row["length"])) for row in rows), [])}

    def data_collator(self, rows):
        maximum = max(len(row["input_ids"]) for row in rows)
        return {"input_ids": [row["input_ids"] + [self.tokenizer.pad_token_id]*(maximum-len(row["input_ids"])) for row in rows],
                "labels": [row["labels"] + [-100]*(maximum-len(row["labels"])) for row in rows]}


@pytest.fixture
def plugin(monkeypatch):
    swift = ModuleType("swift")
    llm = ModuleType("swift.llm")
    template = ModuleType("swift.llm.template")
    registry = {}
    template.Template = SwiftTemplateStub
    template.TemplateMeta = lambda **kwargs: SimpleNamespace(**kwargs)
    template.register_template = lambda meta: registry.setdefault(meta.template_type, meta)
    monkeypatch.setitem(sys.modules, "swift", swift)
    monkeypatch.setitem(sys.modules, "swift.llm", llm)
    monkeypatch.setitem(sys.modules, "swift.llm.template", template)
    path = Path(__file__).resolve().parents[1] / "scripts" / "swift_template.py"
    spec = importlib.util.spec_from_file_location("drift_swift_test_plugin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.registration = registry["drift_chat"]
    return module


def make_template(plugin, **kwargs):
    template = plugin.DriftChatTemplate(ChatTokenizer(), plugin.registration, max_length=kwargs.pop("max_length", 4096), **kwargs)
    template.mode = "train"
    return template


def conversation():
    return [{"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}]


def test_outer_encode_matches_extraction_and_preserves_input(plugin):
    template = make_template(plugin)
    rows = conversation()
    original = deepcopy(rows)
    expected = assistant_labels(template.tokenizer, rows, 4096)
    result = template.encode({"messages": rows}, return_length=True)
    assert result["input_ids"] == expected["input_ids"]
    assert result["labels"] == expected["labels"]
    assert result["length"] == len(expected["input_ids"])
    assert "loss_scale" not in result
    assert "".join(chr(value) for value in result["labels"] if value != -100) == "A1~A2$"
    assert rows == original


@pytest.mark.parametrize("system", ["SYSTEM", ""])
def test_std_inputs_reconstruct_separate_system_once(plugin, system):
    template = make_template(plugin)
    rows = conversation()
    rows[0]["content"] = system
    chosen = SimpleNamespace(messages=deepcopy(rows[1:]), system=system)
    result = template.encode(SimpleNamespace(chosen=chosen))
    expected = assistant_labels(template.tokenizer, rows)
    assert result["input_ids"] == expected["input_ids"]
    assert chosen.messages == rows[1:]
    chosen.messages.insert(0, rows[0])
    assert template._encode(chosen)["input_ids"] == expected["input_ids"]


def test_system_conflict_and_nontext_input_fail(plugin):
    template = make_template(plugin)
    with pytest.raises(ValueError, match="Conflicting system"):
        template._encode(SimpleNamespace(messages=conversation(), system="different"))
    with pytest.raises(ValueError, match="not None"):
        template.encode(None)
    with pytest.raises(ValueError, match="content must be text"):
        template.encode({"messages": [{"role": "assistant", "content": None}]})
    with pytest.raises(ValueError, match="not None"):
        template._encode(None)


@pytest.mark.parametrize("mode", ["pt", "vllm", "lmdeploy", "sglang", "rlhf", "kto"])
def test_non_sft_modes_cannot_bypass_plugin(plugin, mode):
    template = make_template(plugin)
    template.mode = mode
    data = {"messages": conversation()}
    with pytest.raises(ValueError, match="only for causal-LM SFT"):
        template.encode(data)
    with pytest.raises(ValueError, match="only for causal-LM SFT"):
        template._encode_truncated(SimpleNamespace(messages=conversation(), system=None))
    assert data["messages"] == conversation()


@pytest.mark.parametrize("field", ["tools", "images", "audios", "videos", "objects"])
def test_tools_and_multimodal_preprocessing_are_not_allowed(plugin, field):
    template = make_template(plugin)
    chosen = SimpleNamespace(messages=conversation(), system=None, **{field: ["value"]})
    with pytest.raises(ValueError, match="text-only"):
        template._encode_truncated(chosen)


@pytest.mark.parametrize("options", [{"loss_scale": "all"}, {"loss_scale": "last_round"},
    {"template_backend": "jinja"}, {"use_chat_template": False}, {"response_prefix": "PREFIX"},
    {"enable_thinking": True}, {"truncation_strategy": "left"}, {"truncation_strategy": "split"}])
def test_incompatible_configuration_is_not_silently_ignored(plugin, options):
    with pytest.raises(ValueError):
        make_template(plugin, **options)


def test_right_truncation_matches_attribution_and_raise_refuses(plugin):
    rows = conversation()
    template = make_template(plugin, max_length=50, truncation_strategy="right")
    expected = assistant_labels(template.tokenizer, rows, max_length=50)
    result = template.encode({"messages": rows}, return_length=True)
    assert result["input_ids"] == expected["input_ids"]
    assert result["labels"] == expected["labels"]
    assert result["length"] == 50
    with pytest.raises(ValueError, match="truncated signed gradient"):
        make_template(plugin, max_length=50, truncation_strategy="raise").encode({"messages": rows})
    with pytest.raises(ValueError, match="no assistant targets"):
        make_template(plugin, max_length=4, truncation_strategy="right").encode({"messages": rows})


def test_native_packing_and_dynamic_padding_receive_lists_and_lengths(plugin):
    template = make_template(plugin)
    short = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    rows = [template.encode({"messages": value}, return_length=True) for value in (short, conversation())]
    assert plugin.DriftChatTemplate.packing_row is SwiftTemplateStub.packing_row
    assert plugin.DriftChatTemplate.data_collator is SwiftTemplateStub.data_collator
    packed = template.packing_row(rows)
    boundary = rows[0]["length"]
    assert packed["labels"] == rows[0]["labels"] + rows[1]["labels"]
    assert packed["labels"][boundary] == -100
    assert packed["position_ids"][boundary] == 0
    padded = template.data_collator(rows)
    assert len(padded["input_ids"][0]) == len(padded["input_ids"][1])
    assert padded["labels"][0][boundary:] == [-100] * (rows[1]["length"] - boundary)
