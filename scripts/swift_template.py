"""ms-swift external SFT template using exactly DRIFT's assistant-token mask.

Load this file with ``--external_plugins`` and select ``--template drift_chat``.
Use ``--template_backend swift --loss_scale default --truncation_strategy right``.
The template renders the tokenizer's complete chat template itself; neither
Swift's model-specific prompt fragments nor its all-token Jinja loss are used.

This adapter targets Swift's Template/StdTemplateInputs API (checked against
ms-swift 3.12.0.dev0). It is intentionally SFT-only, with ordinary Swift dynamic
padding and packing left to the inherited collator.
"""

from __future__ import annotations

from copy import deepcopy

from swift.llm.template import Template, TemplateMeta, register_template

from drift.io import validate_messages
from drift.masking import assistant_labels


class DriftChatTemplate(Template):
    """Preserve complete conversations and supervise all corpus assistant turns."""

    support_padding_free = True

    def __init__(self, processor, template_meta, default_system=None, max_length=None, **kwargs):
        if default_system is not None or getattr(template_meta, "default_system", None) is not None:
            raise ValueError("drift_chat requires system text in each record, not a default_system override")
        if kwargs.get("loss_scale", "default") != "default":
            raise ValueError("drift_chat requires --loss_scale default; the DRIFT mask supplies token weights")
        if kwargs.get("template_backend", "swift") != "swift":
            raise ValueError("drift_chat requires --template_backend swift")
        if not kwargs.get("use_chat_template", True):
            raise ValueError("drift_chat requires use_chat_template=True")
        if kwargs.get("response_prefix") not in (None, "") or kwargs.get("enable_thinking") is True:
            raise ValueError("drift_chat does not add response/thinking prefixes; preserve them in the original messages")
        if kwargs.get("truncation_strategy", "raise") not in {"right", "raise"}:
            raise ValueError("drift_chat supports only right truncation or raise, matching DRIFT corpus extraction")
        super().__init__(processor, template_meta, default_system, max_length, **kwargs)
        # Swift includes template state in preprocessing-cache fingerprints.
        self._version = "drift-chat-assistant-content-eos-v3"

    def _require_sft(self):
        if self.mode != "train" or self.task_type != "causal_lm":
            raise ValueError("drift_chat is only for causal-LM SFT (template mode='train'); use drift rollout for inference")

    def encode(self, inputs, return_template_inputs=False, return_length=False):
        self._require_sft()
        if inputs is None:
            raise ValueError("drift_chat requires a complete conversation, not None")
        # Swift converts dicts/InferRequests into StdTemplateInputs before
        # _encode_truncated. Reject None/non-text messages before that converter
        # attempts multimodal processing. Do not mutate the caller's object.
        if isinstance(inputs, dict):
            validate_messages(inputs.get("messages"))
        elif hasattr(inputs, "messages"):
            validate_messages(inputs.messages)
        return super().encode(inputs, return_template_inputs=return_template_inputs,
                              return_length=return_length)

    def _conversation(self, inputs):
        if inputs is None:
            raise ValueError("drift_chat requires StdTemplateInputs, not None")
        for key in ("tools", "images", "audios", "videos", "objects"):
            if getattr(inputs, key, None):
                raise ValueError(f"drift_chat supports text-only SFT; {key} must be empty")
        messages = deepcopy(getattr(inputs, "messages", None))
        validate_messages(messages)
        system = getattr(inputs, "system", None)
        if system is not None:
            if not isinstance(system, str):
                raise ValueError("drift_chat system content must be text")
            if messages[0]["role"] == "system":
                if messages[0]["content"] != system:
                    raise ValueError("Conflicting system content in StdTemplateInputs and messages")
            else:
                # Preserve an explicitly empty system message as well: that
                # message may already have been included during attribution.
                messages.insert(0, {"role": "system", "content": system})
        return messages

    def _encode(self, inputs):
        self._require_sft()
        if not getattr(self.tokenizer, "is_fast", False) or not self.tokenizer.chat_template:
            raise ValueError("drift_chat requires a fast tokenizer with its original chat_template")
        if type(self.max_length) is not int or self.max_length < 2:
            raise ValueError("drift_chat requires an explicit max_length >= 2")
        if getattr(self.tokenizer, "truncation_side", "right") != "right":
            raise ValueError("drift_chat requires tokenizer.truncation_side='right', matching corpus extraction")
        if self.truncation_strategy not in {"right", "raise"}:
            raise ValueError("drift_chat supports only right truncation or raise")
        messages = self._conversation(inputs)
        encoded = assistant_labels(self.tokenizer, messages, max_length=self.max_length,
                                   last_only=False, require_complete=self.truncation_strategy == "raise")
        if not encoded["supervised_tokens"]:
            raise ValueError("drift_chat found no assistant targets after truncation")
        labels = list(encoded["labels"])
        # Standalone causal CE never predicts the first token. Keep that same
        # boundary when Swift concatenates independent examples for packing.
        labels[0] = -100
        return {"input_ids": list(encoded["input_ids"]), "labels": labels, "loss_scale": None}

    def _encode_truncated(self, inputs):
        # The base implementation mutates tool/media messages and bypasses a
        # subclass _encode in certain inference modes. This SFT-only adapter
        # instead performs one consistent truncation inside assistant_labels.
        encoded = self._encode(inputs)
        encoded["length"] = len(encoded["input_ids"])
        return encoded


register_template(TemplateMeta(
    template_type="drift_chat",
    prefix=[],
    prompt=["{{QUERY}}"],
    chat_sep=[],
    suffix=[],
    system_prefix=["{{SYSTEM}}"],
    default_system=None,
    template_cls=DriftChatTemplate,
))
