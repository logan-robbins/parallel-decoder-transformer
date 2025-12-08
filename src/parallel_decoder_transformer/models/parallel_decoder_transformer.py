"""Parallel Decoder Transformer wired directly onto GPT-OSS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import nn

from ..data.collator_kd import TwoBranchKDCollatorConfig
from ..inference.dnb_bus import DynamicNotesBus, DynamicNotesBusConfig
from ..inference.snc_cross_attn import SharedNotesCrossAttention, SharedNotesCrossAttentionConfig
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
from .snc_backend import PostTrunkSNC, SNCBackend


@dataclass(slots=True, kw_only=True)
class ParallelDecoderModelConfig:
    hidden_size: int = 4096
    vocab_size: int = 32000
    notes_dim: int = 2048
    num_heads: int = 32
    stream_adapters: StreamAdapterConfig
    notes_bus: DynamicNotesBusConfig
    cross_attention: SharedNotesCrossAttentionConfig
    planner_head: PlannerHeadConfig
    notes_head: NotesHeadConfig
    speculation_head: SpeculationHeadConfig
    agreement_head: AgreementHeadConfig
    coverage_head: CoverageHeadConfig
    stream_classifier_head: StreamClassifierConfig
    plan_vocab_size: int = 65536
    plan_hash_salt: str = "parallel-decoder-v1"
    trunk: TrunkAdapterConfig = field(default_factory=TrunkAdapterConfig)
    instrumentation: Optional[InstrumentationSpec] = None
    collator: TwoBranchKDCollatorConfig = field(
        default_factory=lambda: TwoBranchKDCollatorConfig(pad_token_id=0)
    )


class ParallelDecoderTransformer(nn.Module):
    """Destructive rewrite of the model stack around GPT-OSS."""

    def __init__(
        self,
        config: ParallelDecoderModelConfig,
        *,
        snc_backend: Optional[SNCBackend] = None,
    ) -> None:
        super().__init__()
        self.config = config
        if config.instrumentation is not None and config.instrumentation.enabled:
            adapter_config = InstrumentedTrunkAdapterConfig(
                trunk=config.trunk,
                instrumentation=config.instrumentation,
            )
            self.trunk_adapter: GptOssTrunkAdapter = InstrumentedTrunkAdapter(adapter_config)
        else:
            self.trunk_adapter = GptOssTrunkAdapter(config.trunk)
        self.stream_adapters = StreamAdapters(config.stream_adapters)  # type: ignore[arg-type]
        self.notes_bus = DynamicNotesBus(config.notes_bus)  # type: ignore[arg-type]
        self.cross_attention = SharedNotesCrossAttention(config.cross_attention)  # type: ignore[arg-type]
        if snc_backend is None:
            snc_backend = PostTrunkSNC(self.cross_attention)
        self.snc_backend = snc_backend
        self.planner_head = PlannerHead(config.planner_head)  # type: ignore[arg-type]
        self.notes_head = NotesHead(config.notes_head)  # type: ignore[arg-type]
        self.speculation_head = SpeculationHead(config.speculation_head)  # type: ignore[arg-type]
        self.agreement_head = AgreementHead(config.agreement_head)  # type: ignore[arg-type]
        self.coverage_head = CoverageHead(config.coverage_head)  # type: ignore[arg-type]
        self.stream_classifier = StreamClassifierHead(config.stream_classifier_head)  # type: ignore[arg-type]
        self.plan_embedding = nn.Embedding(config.plan_vocab_size, config.hidden_size)

    def to_trunk_device_and_dtype(self) -> None:
        """Move adapters/heads to the same device and dtype as the trunk.

        This avoids device/dtype mismatches during inference when the trunk is
        materialised on CUDA/MPS and the lightweight heads remain on CPU/FP32.
        The trunk itself is left untouched (it may be sharded by HF accelerate).
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
            self.cross_attention,
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
        trunk = self.trunk_adapter.model
        for param in trunk.parameters():
            return param.dtype
        return torch.float32

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        trunk = self.trunk_adapter.model
        outputs = trunk(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            **generation_kwargs,
        )
        hidden_states = outputs.hidden_states[-1]
        return hidden_states

    def encode_with_notes(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        stream: torch.Tensor | str,
        notes: torch.Tensor,
        notes_mask: Optional[torch.Tensor] = None,
        **generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not isinstance(self.trunk_adapter, InstrumentedTrunkAdapter):
            raise RuntimeError("encode_with_notes requires an instrumented trunk adapter.")
        mask = notes_mask
        target_device = input_ids.device
        if mask is not None:
            mask = mask.to(device=target_device, dtype=torch.bool)
        target_dtype = self.trunk_hidden_dtype()
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
        hidden_states = outputs.hidden_states[-1]
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        stream: torch.Tensor | str,
        notes: torch.Tensor,
        notes_mask: Optional[torch.Tensor] = None,
        plan_item_ids: Optional[torch.Tensor] = None,
        plan_item_mask: Optional[torch.Tensor] = None,
        sectional_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        gate_override: Optional[torch.Tensor] = None
        if sectional_mask is not None:
            mask_tensor = sectional_mask.to(device=hidden_states.device)
            if mask_tensor.dtype not in (torch.bool, torch.uint8):
                mask_tensor = mask_tensor > 0
            if mask_tensor.dim() == 0:
                mask_tensor = mask_tensor.view(1)
            gate_override = mask_tensor.bool()
        # Adapters and SNC are applied within the instrumented trunk
        adapted = hidden_states
        attended = hidden_states

        # Ensure tensors are on the same device as the heads (which may differ from trunk output if sharded)
        # We use planner_head as the reference device for all heads.
        head_device = next(self.planner_head.parameters()).device
        if attended.device != head_device:
            attended = attended.to(head_device)
        if adapted.device != head_device:
            adapted = adapted.to(head_device)

        # Align training with inference by computing planner logits from the
        # notes-conditioned (attended) states. This makes mask-ablation and
        # stability diagnostics meaningful and encourages notes sensitivity.
        planner_logits = self.planner_head(attended)
        notes_logits = self.notes_head(attended)
        speculative_notes = self.speculation_head(adapted)
        agreement_score = self.agreement_head(attended).squeeze(-1)
        stream_logits = self.stream_classifier(adapted)
        coverage_logits: Optional[torch.Tensor] = None
        if plan_item_ids is not None and plan_item_mask is not None:
            # Ensure plan inputs are on the head device
            if plan_item_ids.device != head_device:
                plan_item_ids = plan_item_ids.to(head_device)
            embedded_plan = self.plan_embedding(plan_item_ids)

            if plan_item_mask.device != head_device:
                plan_item_mask = plan_item_mask.to(head_device)
            plan_mask_bool = plan_item_mask.to(dtype=torch.bool)

            coverage_logits = self.coverage_head(attended, embedded_plan, plan_mask_bool)

        # lm_head is part of the trunk, so we must ensure inputs are on ITS device.
        # The trunk output (hidden_states) was likely on the correct device for the next trunk layer/head.
        # If we moved it, we might need to move it back, but 'attended' is a copy/view.
        # However, self.trunk_adapter.model.lm_head might be on a different device from our PDT heads.
        lm_head = self.trunk_adapter.model.lm_head
        lm_head_device = next(lm_head.parameters()).device
        lm_input = attended
        if lm_input.device != lm_head_device:
            lm_input = lm_input.to(lm_head_device)

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
        # If instrumentation is enabled, the trunk adapter manages the active stream adapters
        # and cross-attention instances within its layers. The top-level self.stream_adapters
        # and self.cross_attention are effectively unused "phantoms" and should be ignored.
        is_instrumented = False
        if (
            isinstance(self.trunk_adapter, InstrumentedTrunkAdapter)
            and self.trunk_adapter.instrumentation_enabled
        ):
            is_instrumented = True

        yield from self.trunk_adapter.iter_trainable_parameters()

        if not is_instrumented:
            yield from self.stream_adapters.parameters()
            yield from self.cross_attention.parameters()

        yield from self.planner_head.parameters()
        yield from self.notes_head.parameters()
        yield from self.speculation_head.parameters()
        yield from self.agreement_head.parameters()
        yield from self.coverage_head.parameters()
        yield from self.stream_classifier.parameters()
        yield from self.plan_embedding.parameters()

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        """Collect all trainable parameters for a lightweight checkpoint.

        This captures the exact set of parameters that would be updated by the optimizer,
        ignoring frozen trunk weights but capturing instrumented layer adapters.
        """
        payload: Dict[str, torch.Tensor] = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                payload[name] = param.detach().cpu()
        return payload

    def load_adapters(self, state_dict: Dict[str, torch.Tensor], *, strict: bool = False) -> None:
        """Load adapter/head parameters from a checkpoint.

        Uses strict=False by default to allow loading adapters onto a model with
        frozen/missing trunk keys, but ensures that provided keys match.
        """
        keys = self.load_state_dict(state_dict, strict=False)
        if strict and (keys.missing_keys or keys.unexpected_keys):
            # Filter missing keys to only complain if a *trainable* param is missing
            trainable_names = {n for n, p in self.named_parameters() if p.requires_grad}
            missing_trainable = [k for k in keys.missing_keys if k in trainable_names]

            if missing_trainable or keys.unexpected_keys:
                raise RuntimeError(
                    f"Adapter load failed strict check.\n"
                    f"Missing trainable: {missing_trainable}\n"
                    f"Unexpected: {keys.unexpected_keys}"
                )


__all__ = ["ParallelDecoderTransformer", "ParallelDecoderModelConfig"]
