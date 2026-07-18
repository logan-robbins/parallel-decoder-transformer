# Data directory

Only the pinned-revision historical pipeline is admissible for the current
experiment. The archived `pdt_10k`, short-sentence, QA, and cleaned Hugging Face
Wikipedia datasets are historical artifacts and must not be used to train or
evaluate the real-plan architecture.

## Canonical layout

```text
data/
├── model_intrinsic_parallel/
│   ├── candidates/
│   │   └── audit_round_2_titles.txt
│   ├── source/
│   │   ├── audit_round_2/
│   │   │   ├── blackwater_fire_of_1937.json
│   │   │   ├── battle_of_sluys.json
│   │   │   └── great_stink.json
│   │   ├── audit_round_2_screen.json
│   │   └── baltimore_railroad_strike_1877.json
│   ├── curation/
│   │   ├── baltimore_railroad_strike_1877.json
│   │   ├── blackwater_fire_of_1937.json
│   │   ├── battle_of_sluys.json
│   │   └── great_stink.json
│   └── examples/
│       ├── baltimore_railroad_strike_1877.json
│       ├── blackwater_fire_of_1937.json
│       ├── battle_of_sluys.json
│       └── great_stink.json
├── raw/
│   ├── model_intrinsic_parallel/audit_round_2/
│   │   ├── pinned_revisions.jsonl
│   │   └── logs/
│   ├── historical/<corpus>/
│   │   ├── candidates.jsonl
│   │   └── pinned_revisions.jsonl
│   └── real_plan/<corpus>/
│       ├── all_examples.jsonl
│       └── examples/
│           ├── train.jsonl
│           ├── validation.jsonl
│           ├── test.jsonl
│           └── manifest.json
├── accepted/historical/<corpus>/
│   ├── accepted_sources.jsonl
│   ├── accepted_manifest.json
│   ├── rejections.json
│   └── selection.json
├── teacher/<corpus>/
│   ├── fact_requests.jsonl
│   ├── fact_results.jsonl
│   ├── facts.jsonl
│   ├── joint_requests.jsonl
│   └── joint_results.jsonl
└── processed/real_plan/<trunk-profile>/
    ├── train.jsonl
    ├── validation.jsonl
    └── test.jsonl
```

Raw data is immutable. Derived files are published exclusively and may be
regenerated from their recorded inputs. The scripts refuse to overwrite output
paths and publish multi-file bundles atomically.

`model_intrinsic_parallel` contains four manually curated contract-inspection
records. Their sources are rebuilt directly from immutable pinned Wikimedia
revisions, and each compiled example stores exact fact provenance, explicit
semantic presentation order, exhaustive owner/reference/absent labels, all six
random physical decoder bindings, optional delayed cross-lane evidence,
teacher source/claim/decomposition audits, and actual Qwen token IDs. Note
counts intentionally range from zero through three so the physical
three-decoder target does not require manufactured bus use. These records are
not a teacher-generated corpus and must not be treated as empirical results.

## Historical source gate

The reviewed candidate catalog uses `pdt-historical-candidate-v1`. Acquisition
stores `pdt-wikimedia-revision-bundle-v1` with exact page/revision identity, raw
Action API response, complete wikitext, Parsoid HTML, PageAssessments,
reference records, licenses, and cryptographic hashes.

The prose renderer admits only title, hierarchical content headings, and
ordinary paragraphs. It excludes tables, lists, infoboxes, templates, figures,
captions, galleries, maps, coordinates, citation markers, footnote bodies,
bibliographies, URLs, navigation, category and authority-control blocks,
hatnotes, pronunciation, math, code, edit artifacts, and complete
References/Notes/Citations/Sources/Bibliography/Further reading/External
links/See also/Gallery subtrees, including combined reference headings such as
`Notes, citations and sources`.

Eligibility requires:

- an English main-namespace event or completed process, not a redirect,
  disambiguation page, biography, list, timeline, chronology, index, outline,
  year/decade page, or current event;
- an end date at least 25 years before acquisition and a relevant history
  WikiProject assessment of `FA`, `GA`, `A`, or `B`;
- the complete untruncated body at 3,000–7,000 pinned-Qwen tokens, at least six
  substantive sections, and at least twelve substantive paragraphs;
- at least 30 inline reference occurrences, 15 distinct works, eight explicit
  scholarly/institutional source records, citations on at least 70% of
  substantive paragraphs, and no work above 25% of occurrences;
- no citation, neutrality, dispute, original-research, cleanup, or hoax
  maintenance transclusion.

Near-duplicate and reviewed related-page families are clustered before a
deterministic 90/5/5 family split. Selection requires the exact requested count
in each of eight historical categories and never lowers admission thresholds.
If no candidate is eligible or exact category balance cannot be filled, the
filter exits nonzero after atomically publishing `failure.json` and
`rejections.json`. When some rows are eligible but balance fails, it also
publishes integrity-bound `eligible_sources.jsonl` for inspection and future
candidate-pool work. The absence of `accepted_manifest.json` is the hard
downstream stop signal.

Run acquisition and filtering with the commands in the repository
[README](../README.md). `accepted_manifest.json` hashes the exact accepted
source file, renderer, tokenizer, input raw bundle, sources, and split/family
assignments. Every Batch request must carry an explicitly approved SHA-256 of
that manifest.

## Teacher and processed contracts

The fact stage emits 18–48 cited atomic facts and one contradicted hard negative
per fact. Provenance is an exact paragraph substring plus one or more reference
IDs attached to that paragraph. Facts must span at least twelve paragraphs and
four sections.

The joint stage emits one prompt, exactly three semantic plans with explicit
presentation order, and three 700–1,000-token multi-paragraph sections. Every
positive fact has one semantic owner role in the teacher labels. Physical
`D1`/`D2`/`D3` assignment is randomized over all six permutations. References
are optional; every reference that is present has exact source/receiver
evidence, points backward in presentation order, and has a valid one-block
delay after tokenization.

Validated examples use `pdt-real-plan-v2`. Run
`scripts/prepare_real_plan_data.py split-examples` to publish family-disjoint
train, validation, and test files with a hashed split manifest. Retokenization
uses only locally cached pinned Qwen and BGE assets and emits
`pdt-real-plan-tokenized-v3`. It never downloads or substitutes a tokenizer or
embedding model implicitly.
