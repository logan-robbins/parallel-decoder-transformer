"""Migrate adapters.pt from pre-upgrade-05 format to efficient instrumentation format.

Removes:
  - adapter *.layer_norm.{weight,bias} keys (adapter blocks no longer have LayerNorm)
  - snc_residual.cross_attention.gate keys (SNC internal gate removed from mid-stack path)

Does not touch:
  - down/up projection weights (architecture unchanged)
  - notes_gate / stream_adapter_gate (outer gates preserved)
  - All head weights (agreement, coverage, planner, notes, speculation)
  - PostTrunkSNC cross_attention.gate (only mid-stack SNC gates are removed)
"""
import argparse
from pathlib import Path
import torch

_ADAPTER_LN_SUFFIXES = ("layer_norm.weight", "layer_norm.bias")
_MIDSTACK_SNC_GATE_SUFFIX = "snc_residual.cross_attention.gate"


def migrate(src: Path, dst: Path, dry_run: bool = False) -> None:
    """Strip stale keys from a pre-upgrade-05 adapter checkpoint.

    Args:
        src: Path to the source checkpoint file.
        dst: Path for the migrated output checkpoint.
        dry_run: If True, report what would change without writing a file.
    """
    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    removed: list[str] = []
    retained: list[str] = list(ckpt.keys())

    new_ckpt: dict = {}
    for key, tensor in ckpt.items():
        # Drop adapter LayerNorm params
        if any(key.endswith(suffix) for suffix in _ADAPTER_LN_SUFFIXES):
            removed.append(key)
            continue
        # Drop mid-stack SNC internal gate
        if key.endswith(_MIDSTACK_SNC_GATE_SUFFIX):
            removed.append(key)
            continue
        new_ckpt[key] = tensor

    print(f"Retained {len(new_ckpt)} / {len(retained)} keys")
    print(f"Removed {len(removed)} keys:")
    for k in removed:
        print(f"  - {k}")

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(new_ckpt, dst)
        print(f"Saved migrated checkpoint to {dst}")
    else:
        print("[dry-run] No file written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate adapter checkpoint to upgrade-05 format")
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    migrate(args.src, args.dst, dry_run=args.dry_run)
