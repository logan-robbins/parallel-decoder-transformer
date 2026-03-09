# Worked Example: Planner-Seeded Decomposition for War Queries

This note gives a concrete example for the prompt **"tell me about the civil war"** and explains how PDT handles training and inference with planner-seeded latent slots.

## 1) Example training data shape

Below is a simplified single-example JSONL-style payload (illustrative, not exact schema-complete):

```json
{
  "student_ids": [1, 314, 892, 77, 2, 991, 44, 2],
  "student_labels": [314, 892, 77, 2, 991, 44, 2, -100],
  "planner_ids": [1204, 8821, 4410, -100, -100, -100, -100, -100],
  "agreement_labels": [1, 1, 0, 1],
  "plan_item_ids": [1204, 8821, 4410],
  "plan_item_mask": [true, true, true],
  "plan_item_streams": [0, 1, 2],
  "coverage_targets": [1.0, 0.0, 0.0],
  "coverage_supervision_mask": [true, true, true],
  "metadata": {
    "sectional_independence": true,
    "teacher_plan": {
      "segments": [
        {"stream": "leadup", "paragraph_start": 0},
        {"stream": "battles", "paragraph_start": 1},
        {"stream": "outcome", "paragraph_start": 2}
      ]
    },
    "stream_surface_lengths": {
      "leadup": 180,
      "battles": 210,
      "outcome": 160
    }
  }
}
```

Interpretation:
- `planner_ids` supervise latent slots (fixed slot count; padded slots ignored).
- `plan_item_ids` are canonical latent plan ids used for coverage ownership.
- `coverage_targets` indicate which plan items this stream/block should own.
- `sectional_independence` + segment metadata let the collator mask LM labels so each stream learns its assigned section span.

## 2) War-query decomposition example (conceptual latent slots)

For **"tell me about the civil war"**, a planner could map to latent slots corresponding to:
- S1: lead-up and parties involved
- S2: major battles and notable deaths
- S3: outcome and reconstruction / aftermath

The paper explicitly treats these as **latent plan vocabulary ids** rather than literal output tokens. Snapshot 0 is computed by re-embedding active slot ids and projecting them into notes space.

## 3) Training flow on this input

1. Prompt hidden states are produced by frozen trunk.
2. Planner head predicts slot logits; CE loss trains slot ids.
3. Active slot ids are embedded (`plan_embedding`) and projected (`plan_notes_proj`) to create planner seed (snapshot 0 alignment losses can supervise this path).
4. Teacher notes supervise `notes_head` and `speculation_head` via MSE.
5. Coverage head predicts ownership over canonical plan items.
6. Agreement head predicts continuation readiness labels.
7. LM CE/KD are applied with sectional masks so each stream is trained mainly on its allocated section text.

## 4) Inference flow for a similar query

Query: **"tell me about the peloponnesian war"**

1. Encode full prompt once.
2. Run planner head once on full prompt hidden states.
3. Argmax latent slot ids, embed + project to planner snapshot 0, publish to Dynamic Notes Bus before stream decoding.
4. Each active stream decodes a provisional stride (`tau` tokens) with SNC reads from visible notes window.
5. Each stream writes speculative note summary at stride boundary.
6. Coverage + readiness computed; gate decides commit vs selective rollback/stall.
7. Repeat until done; merge by ownership/section ordering.

## 5) Clarifying the "token-by-token" confusion

The key is **timescale separation**:
- Planner projection is **prompt-time**, before token generation starts.
- Token decoding is then causal/token-by-token, but conditioned on a lagged notes window that already includes planner snapshot 0.
- Commit/rollback is stride-time (block boundaries), not per token.

So the model does not need the "full future output" to plan. It uses the **full input prompt representation** (`H_x`) in a pre-decode pass to seed latent plan state, then decodes token-by-token with that state available via SNC.
