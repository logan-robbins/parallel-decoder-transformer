"""Tests for the upgrade-05 checkpoint migration script."""
import tempfile
from pathlib import Path
import torch
import pytest


def _make_fake_checkpoint() -> dict:
    """Construct a dict that mimics a pre-upgrade-05 adapter checkpoint."""
    return {
        "trunk_adapter.instrumented_layers.0.stream_adapter.adapters.adapters.stream_0.down.weight": torch.randn(4, 8),
        "trunk_adapter.instrumented_layers.0.stream_adapter.adapters.adapters.stream_0.down.bias": torch.randn(4),
        "trunk_adapter.instrumented_layers.0.stream_adapter.adapters.adapters.stream_0.up.weight": torch.randn(8, 4),
        "trunk_adapter.instrumented_layers.0.stream_adapter.adapters.adapters.stream_0.up.bias": torch.randn(8),
        "trunk_adapter.instrumented_layers.0.stream_adapter.adapters.adapters.stream_0.layer_norm.weight": torch.ones(8),
        "trunk_adapter.instrumented_layers.0.stream_adapter.adapters.adapters.stream_0.layer_norm.bias": torch.zeros(8),
        "trunk_adapter.instrumented_layers.0.snc_residual.cross_attention.q_proj.weight": torch.randn(8, 8),
        "trunk_adapter.instrumented_layers.0.snc_residual.cross_attention.gate": torch.tensor([-5.0]),
        "trunk_adapter.instrumented_layers.0.notes_gate": torch.tensor([0.0]),
        "trunk_adapter.instrumented_layers.0.stream_adapter_gate": torch.tensor([0.0]),
    }


def test_migration_removes_layer_norm_keys():
    from scripts.migrate_checkpoint_05 import migrate
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "old.pt"
        dst = Path(tmpdir) / "new.pt"
        torch.save(_make_fake_checkpoint(), src)
        migrate(src, dst, dry_run=False)
        new_ckpt = torch.load(dst, map_location="cpu", weights_only=True)
        assert not any("layer_norm" in k for k in new_ckpt)


def test_migration_removes_snc_internal_gate():
    from scripts.migrate_checkpoint_05 import migrate
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "old.pt"
        dst = Path(tmpdir) / "new.pt"
        torch.save(_make_fake_checkpoint(), src)
        migrate(src, dst, dry_run=False)
        new_ckpt = torch.load(dst, map_location="cpu", weights_only=True)
        assert not any(k.endswith("snc_residual.cross_attention.gate") for k in new_ckpt)


def test_migration_preserves_projection_weights():
    from scripts.migrate_checkpoint_05 import migrate
    old = _make_fake_checkpoint()
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "old.pt"
        dst = Path(tmpdir) / "new.pt"
        torch.save(old, src)
        migrate(src, dst, dry_run=False)
        new_ckpt = torch.load(dst, map_location="cpu", weights_only=True)
    for key in ("down.weight", "down.bias", "up.weight", "up.bias"):
        matching = [k for k in new_ckpt if k.endswith(key)]
        assert matching, f"Key ending in {key!r} must be preserved"


def test_migration_dry_run_writes_no_file():
    from scripts.migrate_checkpoint_05 import migrate
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "old.pt"
        dst = Path(tmpdir) / "new.pt"
        torch.save(_make_fake_checkpoint(), src)
        migrate(src, dst, dry_run=True)
        assert not dst.exists(), "dry_run must not write the output file"
