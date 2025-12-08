"""Tokenizer loading helpers aligned with GPT-OSS training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

try:  # pragma: no cover - optional heavy dependency
    from transformers import AutoTokenizer  # type: ignore
    from transformers.tokenization_utils import PreTrainedTokenizer  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency guard
    AutoTokenizer = None  # type: ignore[assignment]
    PreTrainedTokenizer = object  # type: ignore[assignment]
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


DEFAULT_SPECIAL_TOKENS: Tuple[str, ...] = ("<plan>", "<notes>", "<rollback>", "<commit>")


@dataclass(slots=True)
class TokenizerConfig:
    """Configuration for resolving a tokenizer instance."""

    pretrained_name: str = "openai/gpt-oss-20b"
    special_tokens: Sequence[str] = field(default_factory=lambda: list(DEFAULT_SPECIAL_TOKENS))
    use_fast: bool = True
    padding_side: str = "right"
    truncation_side: str = "right"
    additional_kwargs: Mapping[str, Any] = field(default_factory=dict)
    custom_path: Path | None = None

    def normalized_special_tokens(self) -> Tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []
        for token in self.special_tokens:
            if not token:
                continue
            if token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        return tuple(ordered)


@dataclass(slots=True)
class TokenizerManifest:
    """Serializable manifest describing the tokenizer configuration."""

    source: str
    identifier: str
    tokenizer_class: str
    is_fast: bool
    vocab_size: int
    added_tokens: Tuple[str, ...]
    special_tokens: Tuple[str, ...]
    padding_side: str
    truncation_side: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "identifier": self.identifier,
            "tokenizer_class": self.tokenizer_class,
            "is_fast": self.is_fast,
            "vocab_size": self.vocab_size,
            "added_tokens": list(self.added_tokens),
            "special_tokens": list(self.special_tokens),
            "padding_side": self.padding_side,
            "truncation_side": self.truncation_side,
        }

    def write(self, path: Path) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def resolve_tokenizer(config: TokenizerConfig) -> tuple[PreTrainedTokenizer, TokenizerManifest]:
    """Load a tokenizer according to ``config`` and return it alongside a manifest."""

    if AutoTokenizer is None:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "The optional 'transformers' dependency is required for tokenizer operations. "
            "Install it via `pip install parallel-decoder-transformer[data]`."
        ) from _TRANSFORMERS_IMPORT_ERROR

    load_identifier: str | Path
    source: str
    if config.custom_path is not None:
        load_identifier = config.custom_path
        source = "custom"
    else:
        load_identifier = config.pretrained_name
        source = "pretrained"

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(  # type: ignore[assignment]
        load_identifier,
        use_fast=config.use_fast,
        **dict(config.additional_kwargs),
    )

    tokenizer.padding_side = config.padding_side
    if hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = config.truncation_side  # type: ignore[assignment]
    try:
        max_len = getattr(tokenizer, "model_max_length", 0) or 0
        if max_len and max_len < 65536:
            tokenizer.model_max_length = 65536  # type: ignore[attr-defined]
        elif not max_len:
            tokenizer.model_max_length = 65536  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive len guard
        pass

    special_tokens = config.normalized_special_tokens()
    existing_vocab = set(getattr(tokenizer, "get_vocab", lambda: {})().keys())
    existing_specials = set(getattr(tokenizer, "all_special_tokens", []))
    tokens_to_add = [
        token
        for token in special_tokens
        if token not in existing_specials and token not in existing_vocab
    ]
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

    vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer.get_vocab())))
    manifest = TokenizerManifest(
        source=source,
        identifier=str(load_identifier),
        tokenizer_class=tokenizer.__class__.__name__,
        is_fast=bool(getattr(tokenizer, "is_fast", False)),
        vocab_size=vocab_size,
        added_tokens=tuple(tokens_to_add),
        special_tokens=special_tokens,
        padding_side=str(tokenizer.padding_side),
        truncation_side=str(getattr(tokenizer, "truncation_side", config.truncation_side)),
    )
    return tokenizer, manifest


__all__ = [
    "DEFAULT_SPECIAL_TOKENS",
    "TokenizerConfig",
    "TokenizerManifest",
    "resolve_tokenizer",
]
