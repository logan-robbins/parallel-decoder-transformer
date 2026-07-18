"""Focused tests for the canonical PDT checkpoint contract."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import pdt.checkpoint as checkpoint_module
from pdt.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointCorruptError,
    CheckpointIOError,
    CheckpointMismatchError,
    load_checkpoint,
    resume_checkpoint,
    save_checkpoint,
)
from pdt.training.trainer import PDTTrainer


class _TinyPhysicalDecoder(nn.Module):
    def __init__(self, indices: tuple[int, ...], *, fork_layer: int) -> None:
        super().__init__()
        self.fork_layer = fork_layer
        self.num_decoders = 3
        self.layer_indices = indices
        self.branch_weight = nn.Parameter(torch.randn(3, 3, 3))
        self.plan_attention = nn.Linear(3, 3)
        self.snc = nn.Linear(3, 3)
        self.notes_gate = nn.Parameter(torch.tensor(-4.0))
        self.plan_gate = nn.Parameter(torch.tensor(-3.0))


class _TinyModel:
    def __init__(
        self,
        *,
        base_model: str = "test/tiny-trunk",
        revision: str = "0123456789abcdef",
        layer_indices: tuple[int, ...] = (2, 5),
        fork_layer: int = 2,
        num_decoders: int = 3,
        local_path: str | None = None,
        coordination_source: str = "bus",
    ) -> None:
        self.config = SimpleNamespace(
            trunk=SimpleNamespace(
                base_model=base_model,
                revision=revision,
                local_path=local_path,
            ),
            instrumentation=SimpleNamespace(
                enabled=True,
                target_layers=layer_indices,
                fork_layer=fork_layer,
                coordination_source=coordination_source,
            ),
        )
        self.sidecar = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 3))
        self.physical_decoder = _TinyPhysicalDecoder(
            layer_indices,
            fork_layer=fork_layer,
        )
        self.physical_decoder.num_decoders = num_decoders

    def phi_parameters(self):
        yield from self.physical_decoder.parameters()
        yield from self.sidecar.parameters()


def _training_objects(model: _TinyModel):
    optimizer = torch.optim.AdamW(model.phi_parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    return optimizer, scheduler


def _populate_optimizer(model: _TinyModel, optimizer, scheduler) -> None:
    loss = sum(parameter.square().sum() for parameter in model.phi_parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()


def _assert_module_state_equal(left: nn.Module, right: nn.Module) -> None:
    assert left.state_dict().keys() == right.state_dict().keys()
    for key, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[key])


def _assert_phi_equal(left: _TinyModel, right: _TinyModel) -> None:
    _assert_module_state_equal(left.sidecar, right.sidecar)
    _assert_module_state_equal(left.physical_decoder, right.physical_decoder)


def test_save_and_strict_inference_load_round_trip(tmp_path) -> None:
    torch.manual_seed(7)
    source = _TinyModel()
    source_optimizer, source_scheduler = _training_objects(source)
    _populate_optimizer(source, source_optimizer, source_scheduler)
    path = tmp_path / "nested" / "checkpoint.pt"

    saved = save_checkpoint(
        path,
        source,
        source_optimizer,
        source_scheduler,
        global_step=17,
        stage=2,
    )

    torch.manual_seed(19)
    target = _TinyModel()
    metadata = load_checkpoint(path, target)

    assert saved == metadata
    assert metadata.format_version == CHECKPOINT_FORMAT_VERSION
    assert metadata.identity.base_model == "test/tiny-trunk"
    assert metadata.identity.revision == "0123456789abcdef"
    assert metadata.identity.instrumented_layers == (2, 5)
    assert metadata.identity.coordination_source == "bus"
    assert metadata.identity.fork_layer == 2
    assert metadata.identity.num_decoders == 3
    assert metadata.global_step == 17
    assert metadata.stage == 2
    _assert_phi_equal(source, target)


def test_format_v5_records_physical_decoder_and_optimizer_manifest(tmp_path) -> None:
    model = _TinyModel()
    optimizer, scheduler = _training_objects(model)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, scheduler, global_step=0, stage=0)

    payload = torch.load(path, map_location="cpu", weights_only=True)
    manifest = payload["training"]["optimizer_parameter_manifest"]

    assert CHECKPOINT_FORMAT_VERSION == 5
    assert payload["format_version"] == 5
    assert payload["identity"]["coordination_source"] == "bus"
    assert payload["identity"]["fork_layer"] == 2
    assert payload["identity"]["num_decoders"] == 3
    assert "physical_decoder" in payload["phi"]
    assert len(manifest) == 1
    assert [entry["name"] for entry in manifest[0]][:4] == [
        "physical_decoder.branch_weight",
        "physical_decoder.notes_gate",
        "physical_decoder.plan_gate",
        "physical_decoder.plan_attention.weight",
    ]
    assert manifest[0][0] == {
        "name": "physical_decoder.branch_weight",
        "shape": (3, 3, 3),
        "dtype": "torch.float32",
    }
    assert len(manifest[0]) == len(optimizer.param_groups[0]["params"])


def test_resume_restores_optimizer_and_scheduler(tmp_path) -> None:
    source = _TinyModel()
    source_optimizer, source_scheduler = _training_objects(source)
    _populate_optimizer(source, source_optimizer, source_scheduler)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        source,
        source_optimizer,
        source_scheduler,
        global_step=8,
        stage=1,
    )

    target = _TinyModel()
    target_optimizer, target_scheduler = _training_objects(target)
    assert target_optimizer.state_dict()["state"] == {}
    metadata = resume_checkpoint(path, target, target_optimizer, target_scheduler)

    assert metadata.global_step == 8
    assert metadata.stage == 1
    assert target_optimizer.state_dict()["state"]
    assert target_scheduler.state_dict() == source_scheduler.state_dict()
    assert len(target_optimizer.state_dict()["param_groups"]) == len(
        source_optimizer.state_dict()["param_groups"]
    )
    _assert_phi_equal(source, target)


def test_resume_rejects_reordered_equal_shape_parameters_before_mutation(tmp_path) -> None:
    source = _TinyModel()
    source_optimizer, source_scheduler = _training_objects(source)
    _populate_optimizer(source, source_optimizer, source_scheduler)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        source,
        source_optimizer,
        source_scheduler,
        global_step=8,
        stage=1,
    )

    target = _TinyModel()
    reordered = list(target.phi_parameters())
    reordered[-2], reordered[-1] = reordered[-1], reordered[-2]
    target_optimizer = torch.optim.AdamW(reordered, lr=2e-3)
    target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=2, gamma=0.5)
    before_phi = [parameter.detach().clone() for parameter in target.phi_parameters()]
    before_scheduler = deepcopy(target_scheduler.state_dict())

    with pytest.raises(CheckpointMismatchError, match="Optimizer parameter manifest mismatch"):
        resume_checkpoint(
            path,
            target,
            target_optimizer,
            target_scheduler,
        )

    assert target_optimizer.state_dict()["state"] == {}
    assert target_scheduler.state_dict() == before_scheduler
    for expected, parameter in zip(before_phi, target.phi_parameters()):
        torch.testing.assert_close(expected, parameter)


def test_resume_rejects_same_count_foreign_parameter_before_mutation(tmp_path) -> None:
    source = _TinyModel()
    source_optimizer, source_scheduler = _training_objects(source)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        source,
        source_optimizer,
        source_scheduler,
        global_step=0,
        stage=0,
    )

    target = _TinyModel()
    changed = list(target.phi_parameters())
    changed[-1] = nn.Parameter(torch.zeros_like(changed[-1]))
    target_optimizer = torch.optim.AdamW(changed, lr=2e-3)
    target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=2, gamma=0.5)
    before_phi = [parameter.detach().clone() for parameter in target.phi_parameters()]

    with pytest.raises(
        CheckpointMismatchError,
        match="is not a canonical trainable parameter",
    ):
        resume_checkpoint(
            path,
            target,
            target_optimizer,
            target_scheduler,
        )

    assert target_optimizer.state_dict()["state"] == {}
    for expected, parameter in zip(before_phi, target.phi_parameters()):
        torch.testing.assert_close(expected, parameter)


class _TwoStageCurriculum:
    def __init__(self) -> None:
        self.current_stage = -1

    def determine_stage(self, global_step: int) -> int:
        return 0 if global_step < 5 else 1

    def on_step(self, global_step: int) -> int:
        self.current_stage = self.determine_stage(global_step)
        return self.current_stage


def _trainer_shell(model: _TinyModel, tmp_path) -> PDTTrainer:
    trainer = object.__new__(PDTTrainer)
    trainer.model = model
    trainer.optimizer, trainer.scheduler = _training_objects(model)
    trainer.curriculum = _TwoStageCurriculum()
    trainer.global_step = 0
    trainer.telemetry_dir = tmp_path
    return trainer


def test_trainer_save_and_resume_restore_step_and_curriculum_policy(tmp_path) -> None:
    source = _TinyModel()
    source_trainer = _trainer_shell(source, tmp_path / "source")
    source_trainer.global_step = 8
    source_trainer.curriculum.on_step(8)
    _populate_optimizer(source, source_trainer.optimizer, source_trainer.scheduler)
    source_trainer._save_checkpoint()
    path = source_trainer.telemetry_dir / "checkpoints" / "step_00000008.pt"

    target = _TinyModel()
    target_trainer = _trainer_shell(target, tmp_path / "target")
    metadata = target_trainer.resume_from_checkpoint(path)

    assert metadata.global_step == 8
    assert metadata.stage == 1
    assert target_trainer.global_step == 8
    assert target_trainer.curriculum.current_stage == 1
    assert target_trainer.optimizer.state_dict()["state"]
    _assert_phi_equal(source, target)


def test_trainer_rejects_checkpoint_stage_step_mismatch(tmp_path) -> None:
    source = _TinyModel()
    optimizer, scheduler = _training_objects(source)
    path = tmp_path / "mismatch.pt"
    save_checkpoint(
        path,
        source,
        optimizer,
        scheduler,
        global_step=8,
        stage=0,
    )

    trainer = _trainer_shell(_TinyModel(), tmp_path / "target")
    with pytest.raises(CheckpointMismatchError, match="stage does not match"):
        trainer.resume_from_checkpoint(path)


@pytest.mark.parametrize("mismatch", ["optimizer", "scheduler"])
def test_resume_rejects_training_object_type_mismatch_before_phi_mutation(
    tmp_path, mismatch
) -> None:
    source = _TinyModel()
    source_optimizer, source_scheduler = _training_objects(source)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        source,
        source_optimizer,
        source_scheduler,
        global_step=8,
        stage=1,
    )

    target = _TinyModel()
    target_optimizer, target_scheduler = _training_objects(target)
    before = deepcopy(target.sidecar.state_dict())
    if mismatch == "optimizer":
        target_optimizer = torch.optim.SGD(target.phi_parameters(), lr=1e-2)
    else:
        target_scheduler = torch.optim.lr_scheduler.ExponentialLR(target_optimizer, gamma=0.9)

    with pytest.raises(CheckpointMismatchError, match=f"{mismatch.title()} type mismatch"):
        resume_checkpoint(
            path,
            target,
            target_optimizer,
            target_scheduler,
        )
    for key, value in before.items():
        torch.testing.assert_close(value, target.sidecar.state_dict()[key])


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("base_model", "other/trunk", "base_model mismatch"),
        ("revision", "different-revision", "revision mismatch"),
        ("layer_indices", (2, 8), "Instrumentation layer mismatch"),
        ("coordination_source", "self_only", "Coordination source mismatch"),
        ("fork_layer", 3, "Physical decoder fork mismatch"),
        ("num_decoders", 2, "num_decoders must equal"),
    ],
)
def test_load_rejects_model_identity_mismatch_without_mutating_phi(
    tmp_path, field, replacement, message
) -> None:
    source = _TinyModel()
    optimizer, scheduler = _training_objects(source)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, source, optimizer, scheduler, global_step=1, stage=0)

    kwargs = {field: replacement}
    target = _TinyModel(**kwargs)
    before = deepcopy(target.sidecar.state_dict())
    with pytest.raises(CheckpointMismatchError, match=message):
        load_checkpoint(path, target)

    for key, value in before.items():
        torch.testing.assert_close(value, target.sidecar.state_dict()[key])


def test_checkpoint_contract_rejects_local_trunk_override(tmp_path) -> None:
    local_source = _TinyModel(local_path="/tmp/untracked-trunk")
    local_optimizer, local_scheduler = _training_objects(local_source)
    path = tmp_path / "checkpoint.pt"

    with pytest.raises(CheckpointMismatchError, match="local_path to be None"):
        save_checkpoint(
            path,
            local_source,
            local_optimizer,
            local_scheduler,
            global_step=0,
            stage=0,
        )

    source = _TinyModel()
    optimizer, scheduler = _training_objects(source)
    save_checkpoint(path, source, optimizer, scheduler, global_step=0, stage=0)
    with pytest.raises(CheckpointMismatchError, match="local_path to be None"):
        load_checkpoint(path, _TinyModel(local_path="/tmp/other-trunk"))


def test_load_rejects_missing_physical_and_sidecar_state_keys(tmp_path) -> None:
    source = _TinyModel()
    optimizer, scheduler = _training_objects(source)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, source, optimizer, scheduler, global_step=1, stage=0)
    canonical = torch.load(path, map_location="cpu", weights_only=True)

    missing_physical = deepcopy(canonical)
    missing_physical["phi"]["physical_decoder"].pop("branch_weight")
    missing_physical_path = tmp_path / "missing-physical.pt"
    torch.save(missing_physical, missing_physical_path)
    with pytest.raises(CheckpointMismatchError, match=r"missing=\['branch_weight'\]"):
        load_checkpoint(missing_physical_path, _TinyModel())

    missing_parameter = deepcopy(canonical)
    missing_parameter["phi"]["sidecar"].pop("0.weight")
    missing_parameter_path = tmp_path / "missing-parameter.pt"
    torch.save(missing_parameter, missing_parameter_path)
    with pytest.raises(CheckpointMismatchError, match=r"missing=\['0.weight'\]"):
        load_checkpoint(missing_parameter_path, _TinyModel())


def test_load_rejects_tensor_shape_mismatch(tmp_path) -> None:
    source = _TinyModel()
    optimizer, scheduler = _training_objects(source)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, source, optimizer, scheduler, global_step=1, stage=0)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["phi"]["physical_decoder"]["notes_gate"] = torch.zeros(2)
    torch.save(payload, path)

    with pytest.raises(CheckpointMismatchError, match="notes_gate shape mismatch"):
        load_checkpoint(path, _TinyModel())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("training"), "checkpoint root fields are invalid"),
        (
            lambda payload: payload["training"].update({"global_step": -1}),
            "global_step must be a non-negative integer",
        ),
        (
            lambda payload: payload["training"].update({"stage": 4}),
            r"stage must be an integer in \[0, 3\]",
        ),
        (
            lambda payload: payload["training"]["optimizer"].pop("param_groups"),
            "optimizer fields are invalid",
        ),
        (
            lambda payload: payload["training"].pop("optimizer_parameter_manifest"),
            "training fields are invalid",
        ),
        (
            lambda payload: payload["training"]["optimizer_parameter_manifest"][0][0].pop("dtype"),
            "optimizer_parameter_manifest.*fields are invalid",
        ),
        (
            lambda payload: payload["training"]["optimizer"]["param_groups"][0].pop("params"),
            "params must be a non-empty list",
        ),
        (
            lambda payload: payload["training"]["optimizer"]["param_groups"][0]["params"].append(0),
            "duplicate parameter IDs",
        ),
        (
            lambda payload: payload["training"]["optimizer"]["state"].update({"not-an-id": {}}),
            "must reference a declared integer parameter ID",
        ),
        (
            lambda payload: payload["training"].update({"scheduler": {}}),
            "scheduler must be a non-empty mapping",
        ),
    ],
)
def test_load_validates_complete_training_payload(tmp_path, mutation, message) -> None:
    source = _TinyModel()
    optimizer, scheduler = _training_objects(source)
    canonical_path = tmp_path / "canonical.pt"
    save_checkpoint(canonical_path, source, optimizer, scheduler, global_step=3, stage=1)
    payload = torch.load(canonical_path, map_location="cpu", weights_only=True)
    mutation(payload)
    broken_path = tmp_path / f"broken-{message[:5]}.pt"
    torch.save(payload, broken_path)

    with pytest.raises(CheckpointCorruptError, match=message):
        load_checkpoint(broken_path, _TinyModel())


def test_load_reports_unreadable_and_unsupported_files(tmp_path) -> None:
    unreadable = tmp_path / "unreadable.pt"
    unreadable.write_bytes(b"not a torch checkpoint")
    with pytest.raises(CheckpointCorruptError, match="Could not read checkpoint"):
        load_checkpoint(unreadable, _TinyModel())

    source = _TinyModel()
    optimizer, scheduler = _training_objects(source)
    unsupported = tmp_path / "unsupported.pt"
    save_checkpoint(unsupported, source, optimizer, scheduler, global_step=0, stage=0)
    payload = torch.load(unsupported, map_location="cpu", weights_only=True)
    payload["format_version"] = CHECKPOINT_FORMAT_VERSION + 1
    torch.save(payload, unsupported)
    with pytest.raises(CheckpointMismatchError, match="Unsupported checkpoint format_version"):
        load_checkpoint(unsupported, _TinyModel())


def test_atomic_save_preserves_existing_file_and_cleans_temp_on_failure(
    tmp_path, monkeypatch
) -> None:
    model = _TinyModel()
    optimizer, scheduler = _training_objects(model)
    destination = tmp_path / "checkpoint.pt"
    save_checkpoint(destination, model, optimizer, scheduler, global_step=0, stage=0)
    original = destination.read_bytes()

    def _fail_after_partial_write(_payload, handle) -> None:
        handle.write(b"partial checkpoint")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(checkpoint_module.torch, "save", _fail_after_partial_write)
    with pytest.raises(CheckpointIOError, match="Could not atomically save checkpoint"):
        save_checkpoint(destination, model, optimizer, scheduler, global_step=1, stage=0)

    assert destination.read_bytes() == original
    assert list(tmp_path.glob(".checkpoint.pt.*.tmp")) == []


def test_save_rejects_instantiated_layers_that_disagree_with_config(tmp_path) -> None:
    model = _TinyModel()
    model.physical_decoder.layer_indices = (2,)
    optimizer, scheduler = _training_objects(model)

    with pytest.raises(CheckpointMismatchError, match="do not match model config"):
        save_checkpoint(
            tmp_path / "checkpoint.pt",
            model,
            optimizer,
            scheduler,
            global_step=0,
            stage=0,
        )
