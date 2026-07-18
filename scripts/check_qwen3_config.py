"""Fail-fast checkpoint and tokenizer preflight for the canonical PDT trunk."""

from __future__ import annotations

import argparse

from transformers import AutoConfig, AutoTokenizer

from pdt.config.schemas import DEFAULT_TRUNK_PROFILE, TRUNK_PROFILES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trunk-profile",
        choices=tuple(TRUNK_PROFILES),
        default=DEFAULT_TRUNK_PROFILE,
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the pinned checkpoint metadata to exist in the local HF cache.",
    )
    args = parser.parse_args()
    profile = TRUNK_PROFILES[args.trunk_profile]

    load_kwargs = {
        "revision": profile.revision,
        "local_files_only": args.local_files_only,
    }
    print(f"Checking tokenizer: {profile.base_model}@{profile.revision}")
    tokenizer = AutoTokenizer.from_pretrained(profile.base_model, use_fast=True, **load_kwargs)
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
    config = AutoConfig.from_pretrained(profile.base_model, **load_kwargs)
    expected = {
        "hidden_size": profile.hidden_size,
        "num_hidden_layers": profile.num_hidden_layers,
        "num_attention_heads": profile.num_attention_heads,
        "num_key_value_heads": profile.num_key_value_heads,
    }
    mismatches = {
        name: (getattr(config, name, None), value)
        for name, value in expected.items()
        if getattr(config, name, None) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Checkpoint architecture does not match {profile.name!r}: {mismatches}"
        )

    print(
        "  checkpoint contract passed: "
        f"hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
        f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
        "chat_template=yes"
    )


if __name__ == "__main__":
    main()
