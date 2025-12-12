
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional
import hashlib
import re

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

try:  # Optional: only needed for --strip-training-state / --convert-safetensors
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # Optional: nicer format for HF (recommended)
    from safetensors.torch import save_file as safetensors_save_file
except ImportError:  # pragma: no cover
    safetensors_save_file = None  # type: ignore[assignment]

GCS_BUCKET_BASE = "https://storage.googleapis.com/parallel-decoder-transformer"
ARTIFACTS_MAP = {
    "adapters": {
        "local": "experiments/gpt_oss/adapters_step_50000.pt",
        "remote": f"{GCS_BUCKET_BASE}/checkpoints/gpt-oss-8xH100-50000steps/adapters_step_50000.pt"
    },
    "adapters_latest": {
        "local": "experiments/gpt_oss/adapters.pt",
        "remote": f"{GCS_BUCKET_BASE}/checkpoints/gpt-oss-8xH100-50000steps/adapters.pt"
    },
    "training_logs": {
        "local": "experiments/gpt_oss/training_report.json",
        "remote": f"{GCS_BUCKET_BASE}/checkpoints/gpt-oss-8xH100-50000steps/training_report.json"
    },
     "stages_logs": {
        "local": "experiments/gpt_oss/train_run_stages.json",
        "remote": f"{GCS_BUCKET_BASE}/checkpoints/gpt-oss-8xH100-50000steps/train_run_stages.json"
    },
    "train_manifest": {
        "local": "experiments/gpt_oss/train_manifest.json",
        "remote": f"{GCS_BUCKET_BASE}/checkpoints/gpt-oss-8xH100-50000steps/train_manifest.json"
    },
    "agreement_thresholds": {
        "local": "experiments/gpt_oss/agreement_thresholds.json",
        "remote": f"{GCS_BUCKET_BASE}/checkpoints/gpt-oss-8xH100-50000steps/agreement_thresholds.json"
    }
}

def download_file(url, local_path):
    print(f"Downloading {url} to {local_path}...")
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        # using wget via subprocess to avoid heavy requests/tqdm dep if possible, or fallback to simple curl
        # expecting wget or curl to be present on mac
        if shutil.which("wget"):
             subprocess.run(["wget", url, "-O", local_path], check=True)
        elif shutil.which("curl"):
             subprocess.run(["curl", "-L", url, "-o", local_path], check=True)
        else:
            print("Error: Neither wget nor curl found for downloading artifacts.")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {url}: {e}")
        sys.exit(1)

def _iter_files(directory: Path) -> Iterable[Path]:
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.startswith("."):
                continue
            if "__pycache__" in root:
                continue
            yield Path(root) / filename


def _copytree_filtered(src: Path, dst: Path, *, allow_suffixes: Optional[set[str]] = None) -> int:
    """Copy a directory tree; optionally filter by file suffixes.

    Returns number of files copied.
    """
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in _iter_files(src):
        if allow_suffixes is not None and path.suffix not in allow_suffixes:
            continue
        rel = path.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
        copied += 1
    return copied


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _strip_tex(text: str) -> str:
    """Very small LaTeX -> plain-text best-effort for model cards."""
    # Replace common quote styles
    text = text.replace("``", '"').replace("''", '"')
    # Remove citations like \cite{...}
    text = re.sub(r"\\cite\{[^}]+\}", "", text)
    # Replace \textbf{X} etc with X
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textit\{([^}]*)\}", r"\1", text)
    # Remove simple commands like \ref{...}
    text = re.sub(r"\\ref\{[^}]+\}", "", text)
    # Remove LaTeX escapes for percent
    text = text.replace(r"\%", "%")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _read_arxiv_abstract(arxiv_tex_root: Path) -> Optional[str]:
    abstract_path = arxiv_tex_root / "sections" / "00_abstract.tex"
    if not abstract_path.exists():
        return None
    raw = abstract_path.read_text(encoding="utf-8")
    # Extract between begin/end{abstract} if present.
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", raw, flags=re.S)
    body = m.group(1) if m else raw
    return _strip_tex(body)


def _read_paper_title(arxiv_tex_root: Path) -> Optional[str]:
    main_tex = arxiv_tex_root / "main.tex"
    if not main_tex.exists():
        return None
    raw = main_tex.read_text(encoding="utf-8")
    m = re.search(r"\\title\{([^}]*)\}", raw)
    if not m:
        return None
    return _strip_tex(m.group(1))


def _truncate_str_for_card(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    head = value[: max(0, max_chars - 24)]
    return head + f" … <{len(value)} chars total>"


def _summarize_for_card(value: object, *, max_list: int, max_str: int) -> object:
    if isinstance(value, str):
        return _truncate_str_for_card(value.replace("\n", "\\n"), max_chars=max_str)
    if isinstance(value, list):
        out = [_summarize_for_card(v, max_list=max_list, max_str=max_str) for v in value[:max_list]]
        if len(value) > max_list:
            out.append(f"... <{len(value) - max_list} more items>")
        return out
    if isinstance(value, dict):
        return {k: _summarize_for_card(v, max_list=max_list, max_str=max_str) for k, v in value.items()}
    return value


def _build_notes_artifact_excerpt_for_card(path: Path) -> Optional[dict[str, object]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    # Keep this stable and high-signal for HF README.
    keep: dict[str, object] = {
        "sample_id": obj.get("sample_id"),
        "domain": obj.get("domain"),
        "plan_path": obj.get("plan_path"),
        "sectional_independence": obj.get("sectional_independence"),
        "lag_delta": obj.get("lag_delta"),
        "note_cadence_M": obj.get("note_cadence_M"),
    }

    true_notes = obj.get("true_notes")
    if isinstance(true_notes, list) and true_notes and isinstance(true_notes[0], dict):
        keep["true_notes_example"] = _summarize_for_card(true_notes[0], max_list=3, max_str=220)

    speculative_notes = obj.get("speculative_notes")
    if isinstance(speculative_notes, list) and speculative_notes and isinstance(speculative_notes[0], dict):
        v0 = speculative_notes[0]
        spec_view: dict[str, object] = {
            "variant_id": v0.get("variant_id"),
            "noise_config": v0.get("noise_config"),
            "lag_delta": v0.get("lag_delta"),
        }
        notes = v0.get("notes")
        if isinstance(notes, list) and notes and isinstance(notes[0], dict):
            spec_view["notes_example"] = _summarize_for_card(notes[0], max_list=3, max_str=220)
        keep["speculative_variant_example"] = _summarize_for_card(spec_view, max_list=3, max_str=220)

    versioned = obj.get("versioned_notes")
    if isinstance(versioned, list) and versioned and isinstance(versioned[0], dict):
        s0 = versioned[0]
        snap_view: dict[str, object] = {
            "snapshot_id": s0.get("snapshot_id"),
            "source": s0.get("source"),
            "lag_delta": s0.get("lag_delta"),
            "note_cadence_M": s0.get("note_cadence_M"),
            "ent_count": s0.get("ent_count"),
            "fact_count": s0.get("fact_count"),
        }
        notes = s0.get("notes")
        if isinstance(notes, list) and notes and isinstance(notes[0], dict):
            snap_view["notes_example"] = _summarize_for_card(notes[0], max_list=3, max_str=220)
        keep["versioned_notes_snapshot_0"] = _summarize_for_card(snap_view, max_list=3, max_str=220)

    if "rollback" in obj:
        keep["rollback"] = _summarize_for_card(obj["rollback"], max_list=3, max_str=220)

    return keep


def _default_model_card(*, base_model: str, code_url: str) -> str:
    # Best-effort: include paper metadata and abstract from the local arXiv LaTeX if present.
    arxiv_url = "https://arxiv.org/abs/2512.10054"
    arxiv_tex_root = Path("docs/arxiv_submission")
    paper_title = _read_paper_title(arxiv_tex_root) or (
        "Parallel Decoder Transformer: Model-Internal Parallel Decoding with Speculative Invariance via Note Conditioning"
    )
    paper_abstract = _read_arxiv_abstract(arxiv_tex_root)
    abstract_block = f"\n\n## Abstract (arXiv)\n\n{paper_abstract}\n" if paper_abstract else ""
    gcs_base = "https://storage.googleapis.com/parallel-decoder-transformer/"
    gcs_manifest = "https://storage.googleapis.com/parallel-decoder-transformer/UPLOAD_MANIFEST.md"
    gcs_checkpoints = (
        "https://storage.googleapis.com/parallel-decoder-transformer/checkpoints/"
        "gpt-oss-8xH100-50000steps/"
    )
    gcs_data_archives = "https://storage.googleapis.com/parallel-decoder-transformer/data/archives/"
    wandb_url = "https://wandb.ai/ljrweb-self/parallel-decoder-transformer/runs/fmuea63a"

    # Optional: embed a small, truncated example of a notes artifact.
    example_path = Path("survey_200141_ff0a0b4f.json")
    example_block = ""
    if example_path.exists():
        excerpt = _build_notes_artifact_excerpt_for_card(example_path)
        if excerpt is not None:
            example_json = json.dumps(excerpt, indent=2, ensure_ascii=False)
            example_block = (
                "\n\n## Example: PDT notes artifact (truncated)\n\n"
                "This is a real sample from the dataset pipeline (`survey_200141_ff0a0b4f.json`), "
                "shown with list/string truncation to keep the model card readable.\n\n"
                "```json\n"
                f"{example_json}\n"
                "```\n\n"
                "To reproduce this view locally:\n\n"
                "```bash\n"
                "uv run python scripts/pretty_notes_artifact.py survey_200141_ff0a0b4f.json\n"
                "```\n"
            )

    return f"""---
language:
  - en
license: mit
tags:
  - parallel-decoding
  - speculative-decoding
  - transformers
  - research
  - arxiv
base_model: {base_model}
library_name: transformers
pipeline_tag: text-generation
paper:
  title: "{paper_title}"
  url: {arxiv_url}
---

# Parallel Decoder Transformer (PDT) adapters for GPT-OSS-20B

This repository contains **PDT adapter/head weights** trained against the GPT-OSS-20B trunk, plus minimal training artifacts.

**Paper:** [{paper_title}]({arxiv_url})
{abstract_block}
{example_block}

## How to use

1. Install the reference implementation (runtime + scripts):
   - `{code_url}`
2. Download the base trunk model (`{base_model}`) via Hugging Face (or provide a local path).
3. Download the adapter checkpoint from this repo and point `configs/gpt_oss_transfer_production.yaml` (or CLI flags) at it.

## Artifacts (public GCS)

The complete training artifacts and dataset archives are mirrored publicly in GCS:

- **Bucket root:** `{gcs_base}`
- **Upload manifest (full listing):** `{gcs_manifest}`
- **Training checkpoints:** `{gcs_checkpoints}`
- **Dataset archives:** `{gcs_data_archives}`

## Training logs (Weights & Biases)

- **WandB run:** `{wandb_url}`

## Why the dataset is structured this way

PDT is trained on **streamed, structured supervision** produced by a 5-stage pipeline:

- **Stage 2 (Plans):** a 3-stream decomposition plan is generated for each document.
- **Stage 3 (Notes):** we generate **true notes (teacher)** and **speculative notes (student input)** in a consistent schema:
  - `ENT`: entity table (stable ids)
  - `FACT`: grounded tuples with `evidence_span`
  - `COVERAGE`: plan-item status targets (`covered|partial|missing`)
  - `versioned_notes`: lagged, versioned snapshots mirroring the Dynamic Notes Bus semantics
- **Stage 5 (KD Export):** these artifacts are converted into `kd_*.jsonl` where each line is a **stream-level** training example.

This layout is required to support the **teacher→student curriculum** described in the training guide:

- **Stage 0:** planner/notes-head bootstrap (trunk frozen)
- **Stage 1:** stream adapters + SNC cross-attention bootstrap (speculation frozen; teacher notes forced)
- **Stage 2:** enable speculation + notes-bus usage (teacher-heavy mixing)
- **Stage 3:** train agreement + coverage heads for self-correction/rollback behavior (still trunk frozen)

## Citation

```bibtex
@misc{{robbins2025pdt,
  title={{Parallel Decoder Transformer: Model-Internal Parallel Decoding with Speculative Invariance via Note Conditioning}},
  author={{Robbins, Logan}},
  year={{2025}},
  eprint={{2512.10054}},
  archivePrefix={{arXiv}},
  primaryClass={{cs.AI}},
  url={{https://arxiv.org/abs/2512.10054}}
}}
```

## What’s included

- `pdt_adapters.*`: trainable adapter/head weights (no trunk weights unless you intentionally uploaded them)
- `training_report.json`, `train_run_stages.json`, `train_manifest.json`, `agreement_thresholds.json`

## License

- **This repo (adapters + artifacts)**: MIT.
- **Base model**: `{base_model}` is licensed under Apache-2.0 on Hugging Face (also see its `USAGE_POLICY` there).
- **Reference implementation**: MIT at `{code_url}`.
"""


def _write_gitattributes_for_lfs(dst: Path) -> None:
    # Useful if you later choose a git+LFS push from this staged folder.
    _write_text(
        dst / ".gitattributes",
        "\n".join(
            [
                "*.safetensors filter=lfs diff=lfs merge=lfs -text",
                "*.bin filter=lfs diff=lfs merge=lfs -text",
                "*.pt filter=lfs diff=lfs merge=lfs -text",
                "*.jsonl filter=lfs diff=lfs merge=lfs -text",
                "",
            ]
        ),
    )


def _write_sha256_sums(pkg_dir: Path) -> None:
    """Write SHA256SUMS for all non-dot files in the package."""
    entries: list[tuple[str, str]] = []
    for path in sorted(_iter_files(pkg_dir)):
        rel = path.relative_to(pkg_dir).as_posix()
        if rel.startswith("."):
            continue
        if rel in {"SHA256SUMS"}:
            continue
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        entries.append((h.hexdigest(), rel))
    lines = [f"{digest}  {rel}" for digest, rel in entries]
    _write_text(pkg_dir / "SHA256SUMS", "\n".join(lines) + ("\n" if lines else ""))


def _stage_package(args) -> Path:
    pkg_dir = Path(args.package_dir)
    if args.clean_package and pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # Always stage a model card (README.md in HF model repos).
    if args.hf_readme and Path(args.hf_readme).exists():
        shutil.copy2(args.hf_readme, pkg_dir / "README.md")
    elif getattr(args, "use_repo_readme", False) and Path(args.readme).exists():
        # Explicitly requested: fall back to the repo README as-is (note: no HF YAML metadata).
        shutil.copy2(args.readme, pkg_dir / "README.md")
    else:
        _write_text(
            pkg_dir / "README.md",
            _default_model_card(base_model=args.base_model_id, code_url=args.code_url),
        )

    _write_gitattributes_for_lfs(pkg_dir)

    # Tokenizer at repo root is the most compatible with Transformers auto-loading.
    tok_src = Path(args.tokenizer_dir)
    if tok_src.exists():
        copied = _copytree_filtered(tok_src, pkg_dir, allow_suffixes=None)
        print(f"Staged tokenizer files: {copied} files")
    else:
        print(f"Warning: Tokenizer directory {args.tokenizer_dir} not found. Skipping.")

    # Optional: include trunk weights (usually unnecessary since base model is already on HF).
    if args.include_trunk:
        trunk_src = Path(args.model_dir)
        if trunk_src.exists():
            copied = _copytree_filtered(
                trunk_src,
                pkg_dir,
                allow_suffixes={".safetensors", ".json", ".bin", ".pt", ".model", ".jinja"},
            )
            print(f"Staged trunk files: {copied} files")
        else:
            raise SystemExit(
                f"--include-trunk was set but model dir not found: {args.model_dir}"
            )

    # Adapters / artifacts
    adapter_path = Path(args.adapters_path)
    if not adapter_path.exists() and args.download_missing:
        # Download known artifacts if missing.
        print("\nChecking for missing GCS artifacts...")
        for name, paths in ARTIFACTS_MAP.items():
            if not os.path.exists(paths["local"]):
                if args.dry_run:
                    print(
                        f"[DRY RUN] Would download {name} from {paths['remote']} to {paths['local']}"
                    )
                else:
                    download_file(paths["remote"], paths["local"])

    # Refresh adapter path existence after optional download.
    if not adapter_path.exists():
        print(f"Warning: Adapter checkpoint not found at {adapter_path}. Skipping adapters upload.")
    else:
        if args.strip_training_state:
            if torch is None:
                raise SystemExit(
                    "PyTorch is required for --strip-training-state (pip install torch)."
                )
            ckpt = torch.load(adapter_path, map_location="cpu")
            # Training checkpoints are saved as {"adapters": state_dict, "training_state": ...}
            adapters = ckpt.get("adapters", ckpt)
            if not isinstance(adapters, dict):
                raise SystemExit(
                    f"Unexpected adapter checkpoint format at {adapter_path} (expected dict)."
                )
            if args.convert_safetensors:
                if safetensors_save_file is None:
                    raise SystemExit(
                        "safetensors is required for --convert-safetensors "
                        "(pip install safetensors)."
                    )
                out = pkg_dir / "pdt_adapters.safetensors"
                safetensors_save_file(adapters, str(out))
                print(f"Staged adapters (safetensors): {out}")
            else:
                out = pkg_dir / "pdt_adapters.pt"
                torch.save(adapters, out)
                print(f"Staged adapters (torch): {out}")
        else:
            # Upload the checkpoint exactly as produced by training (includes optimizer state).
            out = pkg_dir / adapter_path.name
            shutil.copy2(adapter_path, out)
            print(f"Staged adapters checkpoint: {out}")

    # Optional lightweight training artifacts
    for artifact in (
        "training_report.json",
        "train_run_stages.json",
        "train_manifest.json",
        "agreement_thresholds.json",
    ):
        src = Path(args.experiments_dir) / "gpt_oss" / artifact
        if src.exists():
            shutil.copy2(src, pkg_dir / artifact)

    _write_sha256_sums(pkg_dir)

    return pkg_dir


def main():
    parser = argparse.ArgumentParser(description="Submit PDT project to HuggingFace Hub.")
    parser.add_argument("--repo-id", required=True, help="Target HF repo ID (e.g., username/model-name)")
    parser.add_argument("--token", help="HF API token (optional, defaults to HF_TOKEN env var)")
    parser.add_argument(
        "--token-file",
        default=None,
        help="Path to a file containing the HF token (recommended vs passing --token).",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Only build the local --package-dir (and optionally download artifacts); do not create/upload to HF.",
    )
    parser.add_argument("--private", action="store_true", help="Create/upload to a private HF repo")
    parser.add_argument("--public", action="store_true", help="Create/upload to a public HF repo (overrides --private)")
    parser.add_argument(
        "--package-dir",
        default="hf_package",
        help="Local staging directory containing the exact files to upload",
    )
    parser.add_argument(
        "--clean-package",
        dest="clean_package",
        action="store_true",
        help="Delete and recreate --package-dir before staging",
    )
    parser.add_argument("--model-dir", default="gpt-oss-20b/original", help="Path to trunk model weights directory")
    parser.add_argument("--tokenizer-dir", default="gpt-oss-20b/tokenizer", help="Path to tokenizer directory")
    parser.add_argument(
        "--adapters-path",
        default="experiments/gpt_oss/adapters_step_50000.pt",
        help="Path to adapter checkpoint produced by training",
    )
    parser.add_argument(
        "--experiments-dir",
        default="experiments",
        help="Path to experiments directory (for optional logs)",
    )
    parser.add_argument("--readme", default="README.md", help="Path to README.md")
    parser.add_argument(
        "--hf-readme",
        default=None,
        help="Optional HF-specific model card to upload as README.md (recommended)",
    )
    parser.add_argument(
        "--use-repo-readme",
        action="store_true",
        help="Use the repository README.md as the HF model card (not recommended).",
    )
    parser.add_argument(
        "--base-model-id",
        default="openai/gpt-oss-20b",
        help="The upstream base model id used for training (for model card metadata)",
    )
    parser.add_argument(
        "--code-url",
        default="https://github.com/logan-robbins/parallel-decoder-transformer",
        help="URL to the reference implementation used to run PDT",
    )
    parser.add_argument(
        "--include-trunk",
        action="store_true",
        help="Also upload the trunk weights (usually unnecessary; base model is already on HF)",
    )
    parser.add_argument(
        "--strip-training-state",
        action="store_true",
        help="Repack adapters to include only trainable params (drops optimizer/scheduler state)",
    )
    parser.add_argument(
        "--convert-safetensors",
        action="store_true",
        help="When used with --strip-training-state, save adapters as .safetensors",
    )
    parser.add_argument("--download-missing", action="store_true", help="Download missing artifacts (adapters/logs) from GCS before upload")
    parser.add_argument("--dry-run", action="store_true", help="Simulate upload without actual execution")
    parser.add_argument(
        "--commit-message",
        default="Upload Parallel Decoder Transformer adapters",
        help="Commit message used for the Hub upload",
    )
    
    args = parser.parse_args()

    # Load env vars
    load_dotenv()
    token = None
    if args.token_file:
        token_path = Path(args.token_file)
        if not token_path.exists():
            raise SystemExit(f"--token-file not found: {args.token_file}")
        token = token_path.read_text(encoding="utf-8").strip()
    token = token or args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    if args.stage_only:
        token = None

    if not token and not args.dry_run and not args.stage_only:
        print("Error: HF_TOKEN not found in environment and not provided via --token.")
        sys.exit(1)
        
    api = HfApi(token=token) if token else None

    print(f"Preparing to submit to HuggingFace Repo: {args.repo_id}")
    # Default to private for safety unless explicitly set public.
    private = True
    if args.public:
        private = False
    elif args.private:
        private = True
    
    if args.stage_only:
        print("[STAGE ONLY] Will not create/upload HF repo.")
    elif args.dry_run:
        print(f"[DRY RUN] Would create repo if not exists (private={private})")
    else:
        try:
            create_repo(
                repo_id=args.repo_id,
                token=token,
                private=private,
                exist_ok=True,
                repo_type="model",
            )
            print(f"Repo {args.repo_id} ready.")
        except Exception as e:
            print(f"Error creating/checking repo: {e}")
            sys.exit(1)

    if args.dry_run:
        print(f"[DRY RUN] Would stage HF package at: {args.package_dir}")
    pkg_dir = _stage_package(args)

    if args.stage_only or args.dry_run:
        staged_files = list(_iter_files(pkg_dir))
        label = "STAGE ONLY" if args.stage_only else "DRY RUN"
        print(f"[{label}] Staged {len(staged_files)} files:")
        for path in staged_files[:50]:
            print(f"  - {path}")
        if len(staged_files) > 50:
            print("  ... and others")
        if args.dry_run:
            print("[DRY RUN] Would upload staged folder as a single commit.")
        else:
            print(f"[STAGE ONLY] Package ready at: {pkg_dir}")
        return

    # Upload the staged folder in one go (keeps HF repo tidy and consistent).
    print(f"\nUploading staged package: {pkg_dir}")
    assert api is not None
    api.upload_folder(
        folder_path=str(pkg_dir),
        path_in_repo=".",
        repo_id=args.repo_id,
        repo_type="model",
        token=token,
        commit_message=args.commit_message,
        ignore_patterns=["**/__pycache__/**", "**/.DS_Store", "**/.git/**"],
    )

    print("\nDone! Uploaded:", args.repo_id)

if __name__ == "__main__":
    main()
