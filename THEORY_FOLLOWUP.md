# Fresh-eye implementation update — 2026-07-16

The causal conclusion below remains right: three streams need not wait for one
another's completed text when their unresolved dependencies fit through an
earlier message. The numerical speed discussion needs a stronger qualifier.

The orchestrator now advances the synchronized frontier with one K-row trunk
call per token round. One frontier-owned KV cache, a physical validity mask,
and per-row logical RoPE positions preserve unequal private prompt and
transition lengths. Stream adapters route by batch row, while producer/kind/lag
metadata makes the notes read explicitly addressed.

At the matmul boundary, three separate BF16 calls may fetch a weight matrix
three times; a packed `[3, hidden] @ [hidden, out]` call can fetch it once. That
weight reuse is the speed mechanism. Private KV-cache traffic still grows with
K and context length, so the speedup need not improve monotonically for ever
longer outputs. Long answers first amortize planning/prefill and later become
increasingly KV-bandwidth-heavy.

The dynamic channel is also finite now: four indices into four 256-entry
product codebooks, exactly 32 capacity bits per producer/write. The bus accepts
only the index tuple and decodes the float note itself, so the reconstructed
256-vector is not an unrestricted side channel.

There are still two empirical results to earn separately:

1. a learned finite-rate message closes dependency loss without full sibling
   text or KV; and
2. packed execution makes a K-frontier round materially cheaper than K
   separate decode calls at matched quality.

`THEORY.md` section 0 gives the corrected information, work--span, and roofline
derivation. A real Qwen3-4B MPS audit crossed the first 32-token boundary with
one `(1,18)` planner call, one `(3,18)` prefill, and 33 `(3,1)` continuation
calls, then emitted one four-index message per stream. This verifies packing
and finite transport, not trained coordination or CUDA speedup. The remainder
of this file is preserved as the original follow-up discussion, with this
update controlling wherever the wording conflicts.

• ## The theory

  The core theory is that multiple autoregressive streams can coordinate without exchanging their text or full KV caches.

  A normal decoder has one causal history:

  > previous tokens → next token

  PDT has three persistent causal histories plus a narrow shared latent workspace:

  > private stream history + delayed sibling notes → next block of tokens

  Each stream remains autoregressive internally. What becomes parallel is the execution of the three streams: they advance together in synchronized 32-token blocks and exchange compressed latent summaries only at block
  boundaries.

  The central hypothesis is:

  > If a token span in stream A depends on state computed privately by stream B, then a learned, low-bandwidth note from B should reduce A’s loss on that specific span once the note becomes visible.

  That is much stronger than merely generating three different outputs. It says the shared bus carries causally useful information.

  ## What we actually built

  For three streams, the execution looks like this:

  shared prompt
       │
       ▼
  VQ planner ──► 3 persistent anchor notes
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       stream 0     stream 1     stream 2
       packed row    packed row    packed row
       own adapter   own adapter   own adapter
          │            │            │
          └──────► shared notes bus ◄┘
                       │
                visible next block

  Each stream has:

  - Its own prompt and private observations.
  - Its own logical KV row and generated-token history inside one packed cache.
  - A stream-specific adapter.
  - The same frozen Qwen3-4B trunk.
  - SNC cross-attention layers that read from the shared notes bus.

  At block m, all three streams read the same frozen bus snapshot and advance independently. Each stream then emits one 256-dimensional note through the shared speculation head. Those three writes are committed
  simultaneously only after the round finishes. They become visible at block m+1.

  Therefore, stream 0 cannot see something stream 1 computed earlier in the same block. The one-block reveal delay is structurally enforced, not merely encouraged during training.

  For K=3, every SNC layer sees a fixed six-note window:

  - Three persistent planner anchor notes.
  - The latest eligible dynamic note from each of the three producers.

  Producer identity always occupies a stable addressed position. Dynamic notes use last-write-wins replacement, so the context stays fixed-size rather than growing with generation.

  ## Does this count as three separate forward passes?

  No in the canonical inference executor; yes only as a logical description of
  three causal continuations.

  The architecture maintains three distinct cached decoder continuations, but
  represents them as rows of one frontier-owned packed KV cache. Their token
  histories and logical RoPE positions do not merge. Cross-stream
  communication occurs only through SNC attending to the notes bus.

  During teacher-forced training, each stream/block is evaluated in one cached
  `[B, 32]` forward after prompt prefill. During free generation, one `[K, 1]`
  call advances all live stream rows and consumes every generated token exactly
  once. Unequal prompt and transition lengths use left padding, a physical
  validity mask, and per-row logical positions.

  A real Qwen3-4B MPS call audit observes one packed prefill and one packed
  continuation call per token round. That proves the physical batching premise,
  but does not by itself prove CUDA wall-clock speedup. The matched CUDA
  comparison against K sequential calls remains an empirical gate.

  So this is:

  - Physically packed across streams at inference.
  - Autoregressive within each stream.
  - Synchronized at communication boundaries.
  - Not “three complete answers for the cost of one decoder pass.”

  ## How the latent notes stay aligned

  There are three different meanings of alignment here.

  ### Structural alignment is guaranteed

  Every note has exactly 256 dimensions. Planner notes and dynamic notes enter the same bus representation space. SNC uses shared learned key/value projections to interpret that space, while the shared speculation head
  produces later notes.

  The model therefore cannot silently pass incompatible note types or variable-shaped objects.

  ### Source and time alignment are guaranteed

  Every bus slot has an explicit producer address. Stream 1’s current note cannot be confused with stream 2’s current note merely because writes arrived in a different order.

  Every dynamic note also has a source block. The visibility rule only admits notes whose source block is old enough under Δ=1. Fixed window positions, producer addresses, block indices, and last-write-wins replacement
  provide the bookkeeping alignment.

  ### Semantic alignment must be learned

  We have not hard-coded dimension 17 to mean “risk” or dimension 42 to mean “priority.” The common semantics should emerge because:

  - All streams share the same trunk.
  - All notes use the same writer.
  - All SNC readers use the same latent coordinate system.
  - Receiving-stream language-model loss backpropagates through SNC into the sibling note that affected it.
  - The privileged same-trunk teacher supplies dependency-token KL targets.
  - Planner anchors establish stable stream roles at snapshot zero.

  That is a credible mechanism for learning a shared latent language, but its semantic success is not yet empirically proven. The 4B MPS smoke test proves shapes, dtype boundaries, cache operation, causal scheduling, and
  parameter connectivity—not that trained notes will contain the intended information.

  That is why the evaluator measures targeted interventions:

  - Zero the bus.
  - Scramble sibling dynamic notes.
  - Cycle one product-code index in one producer’s note from one source block.
  - Compare dependency-token loss against nondependency-token loss.
  - Compare against a parameter-matched module that can read only its own stream.

  The last comparison is now structural rather than post hoc. A separate
  `self_only` checkpoint replaces every bus reader at model construction. Its
  fixed `2K` memory contains only that receiver's prompt tail and latest
  delay-eligible block tail; sibling tensors cannot enter the API. Parameter
  shapes, curriculum, optimizer, losses, data, slot/kind/lag headers, and
  target alignment remain identical. Checkpoint format v3 prevents the bus and
  self-only states from cross-loading. The registered statistic is the ratio
  of each condition's paired dependency-span gate-zero gain, and a ratio at or
  above `0.5` is a null result for the communication claim.

  If the latent space is genuinely aligned, damaging stream 1’s relevant note should selectively damage the annotated tokens in the receiving stream that depend on stream 1. A uniform degradation would instead suggest
  generic capacity or noise sensitivity.

  ## What is novel

  The novelty is not “multiple outputs” or “cross-attention” by itself. Similar pieces exist elsewhere.

  The potentially novel contribution is their particular combination:

  - One frozen causal trunk serving multiple persistent continuations.
  - Strictly private token histories and KV caches.
  - A trainable, fixed-width latent communication bottleneck.
  - Explicit producer addressing.
  - Synchronous block commits with a real reveal delay.
  - Planner-generated initial role notes.
  - Learned dynamic notes instead of exchanged text or full KV state.
  - Span-local causal evaluation designed to prove that sibling information traveled through the bus.
  - A parameter-matched self-only baseline to distinguish communication from additional capacity.

  The scientific claim is therefore not that PDT invented parallel generation. It is:

  > A narrow, delayed, learned latent bus may be sufficient for multiple otherwise independent decoder streams to coordinate on cross-stream dependencies without serially exchanging text or exposing full attention memory.

  The architecture and its self-only falsification control now faithfully
  implement that experiment. The next write is a two-update H100 optimizer
  probe that audits the graph after zero-initialized projections open, followed
  by matched 32-example bus and self-only runs. This does not yet prove the
  hypothesis—that still requires trained CUDA causal results—but the remaining
  uncertainty is empirical rather than an absent control path.

  Summary: we built three private decoder continuations that advance under the same round schedule, cross-attend only to a fixed addressed notes bus, and publish synchronized latent updates for the next round. Mechanical
  alignment and causality are enforced; useful semantic alignment is the key empirical claim still to be demonstrated.


› "What it proves locally is that there is no causal dependency requiring stream 0’s text generation to finish before stream 1 can advance within the same round." this is importan to me for breaking linearity. Still seems
  like if 3 things are decoding 3 paragraphs at the same time, you could possibley get 3x throuput (assuming memory / compute not issue) and roughly 2.x speedup?


• Yes—that is the intended break in linearity.

  Suppose a conventional decoder must produce three 1,000-token paragraphs sequentially. Its causal critical path is approximately:

  [
  1000 + 1000 + 1000 = 3000\text{ token steps}
  ]

  With three PDT streams, each paragraph remains internally autoregressive, but corresponding token steps can execute together:

  [
  \max(1000,1000,1000)=1000\text{ synchronized token steps}
  ]

  Every 32 tokens, the streams publish notes, synchronize, and continue using the newly visible notes. They do not wait for another paragraph to finish.

  ## Expected speedup

  For equal-length streams, a useful approximation is:

  [
  \text{speedup}\approx\frac{3t_1}{t_3+t_{\text{bus}}}
  ]

  Here:

  - (t_1) is the cost of one single-stream decoding step.
  - (t_3) is the cost of advancing all three streams together.
  - (t_{\text{bus}}) includes SNC and block-boundary overhead.

  Examples:

  - If a three-stream step costs 1.2× a single-stream step, speedup is approximately 2.5×.
  - If it costs 1.5×, speedup is approximately 2×.
  - With genuinely separate compute resources, it could approach 3×.
  - If the GPU is already saturated and a three-stream step costs nearly 3×, there is little wall-clock speedup, although the architecture remains causally parallel.

  So your expectation of roughly 2.x is entirely plausible. It is not guaranteed until measured on the optimized CUDA execution path.

  ## Throughput versus latency

  It potentially improves both:

  - Throughput: three output tokens are advanced per synchronized decoding step instead of one.
  - Latency for the complete three-paragraph result: determined primarily by the longest stream rather than the sum of all three stream lengths.

  The key comparison is:

  [
  T_{\text{serial}}\propto L_0+L_1+L_2
  ]

  versus:

  [
  T_{\text{PDT}}\propto \max(L_0,L_1,L_2)+\text{synchronization overhead}
  ]

  Stream imbalance matters. If the lengths are 1,000, 1,000, and 3,000 tokens, the long stream becomes the critical path and the speedup is lower.

  ## What must change to realize it physically

  The current code stacks the three logical streams into one physical batch at
  every decoding position. Three independent compute-stream calls are not the
  canonical path because they do not guarantee shared weight traffic.

  Conceptually:

  token step 0:  [stream 0, stream 1, stream 2] → one batched GPU call
  token step 1:  [stream 0, stream 1, stream 2] → one batched GPU call
  ...
  token step 31: [stream 0, stream 1, stream 2] → one batched GPU call
  boundary:      publish and atomically commit three notes
  token step 32: all streams read the new bus snapshot

  This appears as one batched model invocation over three causal decoder lanes
  with private logical histories and one batch-shaped cache.

  ## The fundamental limitation

  PDT can break incidental serialization, but not real informational dependencies.

  If paragraph 3 requires the exact completed text of paragraph 1, paragraph 3 cannot safely finish first without speculation and correction. But if paragraph 3 only needs a compact fact, decision, topic, or state computed
  by stream 1, stream 1 can publish that state through the bus after block 0 and both can continue concurrently.

  That is the real architectural bet:

  > Most useful cross-section dependencies may be much smaller than the full text used to express them.

  If that is true, the notes bus carries those dependencies while all three expensive token-generation processes remain concurrent.

  Summary: the causal path changes from the sum of three paragraph lengths to approximately their maximum. With balanced streams and sufficient GPU parallel capacity, approximately 2–3× throughput and end-to-end latency
  improvement is a credible target; the CUDA benchmark must determine where within that range the implementation lands.


› That is why this probably good with long answer and specifically world / training knowledge that has likely objective "facts" ? doesnt nescessarily need to be crhonological, correct?


• Correct. PDT is best suited to long answers whose information structure is wide rather than deeply sequential.

  A conventional decoder linearizes everything:

  section A → section B → section C

  even when B does not truly depend on A’s completed wording. PDT tries to execute the underlying dependency graph more honestly:

               shared question
              /       |       \
         evidence   mechanism   implications
              \       |       /
               coordinated synthesis

  Good candidates include:

  - Multi-aspect factual reports.
  - Scientific explanations split into theory, evidence, and limitations.
  - Comparisons across several entities or approaches.
  - Technical documentation covering architecture, operation, and evaluation.
  - Answers assembled from independent sources or retrieved documents.
  - Analyses where streams need to agree on a few facts, terms, or conclusions but not exchange entire paragraphs.

  Objective or stable facts are especially compatible because the coordination message can be compact: an entity, date, measurement, conclusion, confidence, or chosen interpretation. The bus need not transmit the complete
  prose used to explain that fact.

  Chronology is not required. Each stream remains chronological within its own paragraph, but the paragraphs themselves can be produced concurrently whenever their substantive dependencies allow it.

  The weaker cases are tasks with a deep dependency chain:

  - Step-by-step mathematical proofs.
  - Stories where every event depends on the previous event.
  - Code where later functions depend on exact earlier implementation choices.
  - Arguments where section three must respond to the precise wording of section two.

  Even those may contain parallelizable subproblems, but their irreducibly sequential core limits the speedup.

  One qualification: “stored in training knowledge” does not make a fact objectively correct. PDT could retrieve three hallucinations in parallel just as easily as one. Its strongest factual form would combine parallel
  streams with supplied evidence or retrieval, using the notes bus to share verified conclusions, citations, entities, and confidence.

  So yes: the natural target is a long, fact-dense answer with broad parallel branches and a relatively shallow coordination graph. PDT exploits the width of the reasoning problem; it does not eliminate its genuine
  sequential depth.
