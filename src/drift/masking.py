"""Assistant-only labels from the complete, context-aware chat template."""
from __future__ import annotations

from .io import validate_messages


def assistant_labels(tokenizer, messages, max_length=4096, *, last_only=False, require_complete=False):
    """Token-mean causal CE targets, including assistant EOS but excluding headers.

    Locate each response in the full conversation, preserving the template
    choice of intermediate versus final end-of-turn markers. Require unchanged
    content and an explicit special-token terminator; never guess token IDs.
    """
    validate_messages(messages)
    if max_length < 2:
        raise ValueError("max_length must be at least two")
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if last_only and messages[-1]["role"] != "assistant":
        raise ValueError("A rollout anchor must end in its sampled assistant response")
    if require_complete:
        complete_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        if len(complete_ids) > max_length:
            raise ValueError("Rollout anchor exceeds attribution context; refusing a truncated signed gradient")
    encoded = tokenizer(text, add_special_tokens=False, truncation=True,
                        max_length=max_length, return_offsets_mapping=True)
    last_character = encoded["offset_mapping"][-1][1] if encoded["offset_mapping"] else 0
    spans = []
    special_tokens = set(getattr(tokenizer, "all_special_tokens", []))
    special_tokens.update(str(value) for value in getattr(tokenizer, "added_tokens_decoder", {}).values()
                          if getattr(value, "special", False))
    special_tokens = sorted((value for value in special_tokens if value), key=len, reverse=True)
    marker = "DRIFT_CONTENT_BOUNDARY_48b370f0"
    if marker in text:
        raise ValueError("Reserved mask sentinel occurs in text")
    for i, message in enumerate(messages):
        if message["role"] != "assistant" or (last_only and i != len(messages)-1):
            continue
        marked_messages = [dict(value) for value in messages]
        marked_messages[i]["content"] = marker
        marked = tokenizer.apply_chat_template(marked_messages, tokenize=False, add_generation_prompt=False)
        if marked.count(marker) != 1:
            raise ValueError("Template transforms assistant content; use a supported template")
        prefix, suffix = marked.split(marker)
        if text != prefix + message["content"] + suffix:
            raise ValueError("Template transforms assistant content or its surrounding conversation")
        if len(prefix) >= last_character:
            break
        # OLMo3 uses different terminators for historical and final replies.
        # Derive the actual terminator from this full-context render, including
        # any whitespace before it, but excluding separators/next-turn headers.
        stripped = suffix.lstrip()
        terminator = next((token for token in special_tokens if stripped.startswith(token)), None)
        if terminator is None:
            raise ValueError("Assistant response needs an explicit special-token terminator")
        stop = len(prefix) + len(message["content"]) + len(suffix)-len(stripped) + len(terminator)
        if last_only and i:
            generated_prefix = tokenizer.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True)
            if generated_prefix != prefix:
                raise ValueError("Anchor training template differs from its generation prompt")
        spans.append((len(prefix), stop))
    ids = encoded["input_ids"]
    labels = [-100] * len(ids)
    for index, (start, end) in enumerate(encoded["offset_mapping"]):
        if end > start and any(start >= a and end <= b for a, b in spans):
            labels[index] = ids[index]
    expected = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                             truncation=True, max_length=max_length)
    if ids != expected:
        raise ValueError("Rendered-text tokenization differs from the official chat template")
    supervised = sum(label != -100 for label in labels[1:])
    return {"input_ids": ids, "labels": labels, "supervised_tokens": supervised}
