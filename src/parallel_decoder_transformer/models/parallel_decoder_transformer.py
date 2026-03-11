"""Parallel Decoder Transformer wired directly onto GPT-OSS.

This is the production model architecture after a destructive rewrite.
The heavy GPT-OSS trunk (possibly sharded by HF Accelerate) is wrapped
in an optional InstrumentedTrunkAdapter that injects stream + notes
conditioning *inside* every transformer layer via SNC (Shared Notes
Cross-attention). All auxiliary heads (planner, notes, speculation,
agreement, coverage, stream classifier) and the language-model head
run in parallel on the final hidden states.

Benefits:
- Single trunk forward pass (no separate adapter/SNC stage)
- Perfect training/inference alignment
- Lightweight adapter checkpoints (~few hundred MB)
- Full support for mixed-device training (trunk on CUDA, heads on CPU)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import nn

from ..data.collator_kd import TwoBranchKDCollatorConfig
from ..integration.gpt_oss import GptOssTrunkAdapter, TrunkAdapterConfig
from ..integration.instrumentation import (
    InstrumentedTrunkAdapter,
    InstrumentedTrunkAdapterConfig,
    InstrumentationSpec,
)
from .heads import (
    AgreementHead,
    AgreementHeadConfig,
    CoverageHead,
    CoverageHeadConfig,
    NotesHead,
    NotesHeadConfig,
    PlannerHead,
    PlannerHeadConfig,
    StreamClassifierConfig,
    StreamClassifierHead,
    SpeculationHead,
    SpeculationHeadConfig,
)
from .stream_adapters import StreamAdapterConfig, StreamAdapters


@dataclass(slots=True, kw_only=True)
class ParallelDecoderModelConfig:
    """Top-level configuration for the ParallelDecoderTransformer.

    Sub-configs are auto-constructed from top-level dimensions when
    left as None. This guarantees that all components share consistent
    hidden_size / notes_dim / vocab_size values.
    """

    hidden_size: int = 2880
    notes_dim: int = 1536

    # Sub-configs (auto-filled in __post_init__)
    stream_adapters: Optional[StreamAdapterConfig] = None
    planner_head: Optional[PlannerHeadConfig] = None
    notes_head: Optional[NotesHeadConfig] = None
    speculation_head: Optional[SpeculationHeadConfig] = None
    agreement_head: Optional[AgreementHeadConfig] = None
    coverage_head: Optional[CoverageHeadConfig] = None
    stream_classifier_head: Optional[StreamClassifierConfig] = None

    # Planning vocabulary (shared with collator and planner head)
    plan_vocab_size: int = 65536
    plan_hash_salt: str = "parallel-decoder-v1"

    # Trunk + optional deep instrumentation
    trunk: TrunkAdapterConfig = field(default_factory=TrunkAdapterConfig)
    instrumentation: Optional[InstrumentationSpec] = None

    # Data collator (kept in sync automatically)
    collator: TwoBranchKDCollatorConfig = field(
        default_factory=lambda: TwoBranchKDCollatorConfig(pad_token_id=0)
    )

    def __post_init__(self) -> None:
        if self.stream_adapters is None:
            self.stream_adapters = StreamAdapterConfig(hidden_size=self.hidden_size)

        if self.planner_head is None:
            self.planner_head = PlannerHeadConfig(
                hidden_size=self.hidden_size,
                vocab_size=self.plan_vocab_size,
            )
        elif self.planner_head.vocab_size != self.plan_vocab_size:
            raise ValueError(
                "planner_head.vocab_size must match plan_vocab_size so planner logits "
                "and plan_embedding share the same latent vocabulary."
            )

        if self.notes_head is None:
            self.notes_head = NotesHeadConfig(
                hidden_size=self.hidden_size,
                notes_dim=self.notes_dim,
            )
        if self.speculation_head is None:
            self.speculation_head = SpeculationHeadConfig(
                hidden_size=self.hidden_size,
                notes_dim=self.notes_dim,
            )
        if self.agreement_head is None:
            self.agreement_head = AgreementHeadConfig(hidden_size=self.hidden_size)
        if self.coverage_head is None:
            self.coverage_head = CoverageHeadConfig(hidden_size=self.hidden_size)
        if self.stream_classifier_head is None:
            self.stream_classifier_head = StreamClassifierConfig(
                hidden_size=self.hidden_size,
                num_streams=len(self.stream_adapters.streams),
            )

        # Keep collator in perfect sync with model (required for training)
        if self.collator.notes_dim != self.notes_dim:
            self.collator.notes_dim = self.notes_dim
        if self.collator.plan_hash_buckets != self.plan_vocab_size:
            raise ValueError(
                "collator.plan_hash_buckets must match plan_vocab_size for canonical "
                "latent planning."
            )
        if self.collator.plan_hash_salt != self.plan_hash_salt:
            raise ValueError(
                "collator.plan_hash_salt must match plan_hash_salt for canonical "
                "latent planning."
            )
        if self.collator.planner_slots != self.planner_head.num_slots:
            raise ValueError(
                "collator.planner_slots must match planner_head.num_slots for "
                "fixed-slot planning."
            )


class ParallelDecoderTransformer(nn.Module):
    """Production Parallel Decoder wired directly onto GPT-OSS.

    Architecture after the destructive rewrite:
      1. (Optional) InstrumentedTrunkAdapter injects stream + notes
         conditioning inside every transformer layer (SNC).
      2. encode() / encode_with_notes() run the trunk once.
      3. forward() runs all lightweight heads in parallel on the
         final hidden states (including the original lm_head).

    Non-instrumented mode is kept only for backward compatibility;
    the recommended path is always instrumentation=True.
    """

    def __init__(self, config: ParallelDecoderModelConfig) -> None:
        super().__init__()
        self.config = config

        # Trunk (possibly sharded) + optional deep instrumentation
        if config.instrumentation is not None and config.instrumentation.enabled:
            adapter_config = InstrumentedTrunkAdapterConfig(
                trunk=config.trunk,
                instrumentation=config.instrumentation,
            )
            self.trunk_adapter: GptOssTrunkAdapter = InstrumentedTrunkAdapter(adapter_config)
        else:
            self.trunk_adapter = GptOssTrunkAdapter(config.trunk)

        # Lightweight components (all moved to trunk device/dtype later)
        self.stream_adapters = StreamAdapters(config.stream_adapters)  # type: ignore[arg-type]
        self.planner_head = PlannerHead(config.planner_head)  # type: ignore[arg-type]
        self.notes_head = NotesHead(config.notes_head)  # type: ignore[arg-type]
        self.speculation_head = SpeculationHead(config.speculation_head)  # type: ignore[arg-type]
        self.agreement_head = AgreementHead(config.agreement_head)  # type: ignore[arg-type]
        self.coverage_head = CoverageHead(config.coverage_head)  # type: ignore[arg-type]
        self.stream_classifier = StreamClassifierHead(config.stream_classifier_head)  # type: ignore[arg-type]

        # Shared planning vocabulary embedding (used only by coverage head)
        self.plan_embedding = nn.Embedding(config.plan_vocab_size, config.hidden_size)

    def to_trunk_device_and_dtype(self) -> None:
        """Move all adapters and heads to the same device/dtype as the trunk.

        This eliminates device mismatches during inference when the trunk
        lives on CUDA/MPS while the lightweight heads would otherwise stay
        on CPU/FP32. The trunk itself is deliberately left untouched
        (it may be sharded by HF Accelerate).
        """
        trunk = self.trunk_adapter.model
        try:
            ref_param = next(trunk.parameters())
        except StopIteration:
            raise RuntimeError(
                "Cannot determine trunk device and dtype because it has no parameters."
            )

        device = ref_param.device
        dtype = ref_param.dtype

        modules = (
            self.stream_adapters,
            self.planner_head,
            self.notes_head,
            self.speculation_head,
            self.agreement_head,
            self.coverage_head,
            self.stream_classifier,
            self.plan_embedding,
        )
        for module in modules:
            module.to(device=device, dtype=dtype)

    def trunk_hidden_dtype(self) -> torch.dtype:
        """Return the dtype of the trunk's parameters (fallback float32)."""
        trunk = self.trunk_adapter.model
        for param in trunk.parameters():
            return param.dtype
        return torch.float32

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        """Plain trunk forward pass (no notes conditioning)."""
        trunk = self.trunk_adapter.model
        outputs = trunk(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            **generation_kwargs,
        )
        return outputs.hidden_states[-1]

    def encode_with_notes(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        stream: torch.Tensor | str,
        notes: torch.Tensor,
        notes_mask: Optional[torch.Tensor] = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        """Trunk forward pass with deep notes/stream conditioning.

        Requires an InstrumentedTrunkAdapter. Conditioning is injected
        directly into every transformer layer via the SNC mechanism.
        """
        if not isinstance(self.trunk_adapter, InstrumentedTrunkAdapter):
            raise RuntimeError("encode_with_notes requires an instrumented trunk adapter.")

        target_device = input_ids.device
        target_dtype = self.trunk_hidden_dtype()

        mask = notes_mask.to(device=target_device, dtype=torch.bool) if notes_mask is not None else None
        payload_notes = notes.to(dtype=target_dtype, device=target_device)

        with self.trunk_adapter.activate_context(
            stream=stream,
            notes=payload_notes,
            notes_mask=mask,
        ):
            outputs = self.trunk_adapter.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                **generation_kwargs,
            )
        return outputs.hidden_states[-1]

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        plan_item_ids: Optional[torch.Tensor] = None,
        plan_item_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all parallel heads on pre-extracted hidden states.

        This is the core of the "parallel decoder" design. Conditioning
        (stream + notes) already happened inside the trunk via
        encode_with_notes(). All heads run on the same activations for
        perfect training/inference alignment.
        """
        # Move to the device of the heads (planner_head is our reference)
        head_device = next(self.planner_head.parameters()).device
        if hidden_states.device != head_device:
            hidden_states = hidden_states.to(head_device)

        # Planner uses notes-conditioned states (training/inference alignment)
        planner_attention_mask = (
            attention_mask.to(head_device) if attention_mask is not None else None
        )
        planner_logits = self.planner_head(hidden_states, attention_mask=planner_attention_mask)

        notes_logits = self.notes_head(hidden_states)
        speculative_notes = self.speculation_head(hidden_states)
        agreement_score = self.agreement_head(hidden_states).squeeze(-1)
        stream_logits = self.stream_classifier(hidden_states)

        coverage_logits: Optional[torch.Tensor] = None
        if plan_item_ids is not None and plan_item_mask is not None:
            plan_item_ids = plan_item_ids.to(head_device)
            plan_item_mask = plan_item_mask.to(head_device)

            embedded_plan = self.plan_embedding(plan_item_ids)
            plan_mask_bool = plan_item_mask.to(dtype=torch.bool)

            coverage_logits = self.coverage_head(
                hidden_states, embedded_plan, plan_mask_bool
            )

        # lm_head lives inside the trunk (may be on a different device)
        lm_head = self.trunk_adapter.model.lm_head
        lm_head_device = next(lm_head.parameters()).device
        lm_input = hidden_states.to(lm_head_device) if hidden_states.device != lm_head_device else hidden_states
        lm_logits = lm_head(lm_input)

        return {
            "planner_logits": planner_logits,
            "notes_logits": notes_logits,
            "speculative_notes": speculative_notes,
            "agreement": agreement_score,
            "stream_logits": stream_logits,
            "coverage_logits": coverage_logits,
            "lm_logits": lm_logits,
        }

    def iter_trainable_parameters(self):
        """Yield exactly the parameters that should be optimized.

        In instrumented mode the trunk_adapter already owns the active
        stream adapters and cross-attention layers; the top-level
        phantoms are ignored.
        """
        yield from self.trunk_adapter.iter_trainable_parameters()

        # Non-instrumented legacy path only
        if not (
            isinstance(self.trunk_adapter, InstrumentedTrunkAdapter)
            and self.trunk_adapter.instrumentation_enabled
        ):
            yield from self.stream_adapters.parameters()

        yield from self.planner_head.parameters()
        yield from self.notes_head.parameters()
        yield from self.speculation_head.parameters()
        yield from self.agreement_head.parameters()
        yield from self.coverage_head.parameters()
        yield from self.stream_classifier.parameters()
        yield from self.plan_embedding.parameters()

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return a lightweight checkpoint containing only trainable parameters.

        Perfect for adapter-only saving/loading (ignores frozen trunk weights).
        """
        payload: Dict[str, torch.Tensor] = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                payload[name] = param.detach().cpu()
        return payload

    def load_adapters(self, state_dict: Dict[str, torch.Tensor], *, strict: bool = False) -> None:
        """Load adapter/head parameters from a checkpoint.

        Uses strict=False by default to support frozen/missing trunk keys.
        When strict=True we only error on missing *trainable* parameters.
        """
        keys = self.load_state_dict(state_dict, strict=False)
        if strict and (keys.missing_keys or keys.unexpected_keys):
            trainable_names = {n for n, p in self.named_parameters() if p.requires_grad}
            missing_trainable = [k for k in keys.missing_keys if k in trainable_names]

            if missing_trainable or keys.unexpected_keys:
                raise RuntimeError(
                    f"Adapter load failed strict check.\n"
                    f"Missing trainable: {missing_trainable}\n"
                    f"Unexpected: {keys.unexpected_keys}"
                )


__all__ = ["ParallelDecoderTransformer", "ParallelDecoderModelConfig"]
