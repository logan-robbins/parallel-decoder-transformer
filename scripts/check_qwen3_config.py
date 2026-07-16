"""Fail-fast checkpoint and tokenizer preflight for the canonical PDT trunk."""

from __future__ import annotations

import argparse

from transformers import AutoConfig, AutoTokenizer


CANONICAL_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
CANONICAL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=CANONICAL_MODEL)
    parser.add_argument("--revision", default=CANONICAL_REVISION)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the pinned checkpoint metadata to exist in the local HF cache.",
    )
    args = parser.parse_args()

    load_kwargs = {
        "revision": args.revision,
        "local_files_only": args.local_files_only,
    }
    print(f"Checking tokenizer: {args.model}@{args.revision}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, **load_kwargs)
    if not tokenizer.chat_template:
        raise RuntimeError(
            "Canonical PDT training requires an instruction tokenizer with a chat template."
        )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "PDT preflight"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if "assistant" not in rendered:
        raise RuntimeError("Tokenizer chat template did not render an assistant generation turn.")

    print("Checking model config...")
    config = AutoConfig.from_pretrained(args.model, **load_kwargs)
    expected = {
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    }
    mismatches = {
        name: (getattr(config, name, None), value)
        for name, value in expected.items()
        if getattr(config, name, None) != value
    }
    if mismatches:
        raise RuntimeError(f"Checkpoint architecture does not match PDT Qwen3-4B: {mismatches}")

    print(
        "  checkpoint contract passed: "
        f"hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
        f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
        "chat_template=yes"
    )


if __name__ == "__main__":
    main()
