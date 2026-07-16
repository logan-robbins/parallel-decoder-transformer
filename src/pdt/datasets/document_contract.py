"""Canonical long-form private-stream document constants."""

from __future__ import annotations


DOCUMENT_CONTRACT_VERSION = "long-form-private-stream-v1"
DOCUMENT_BLOCKS = 32
DOCUMENT_BLOCK_TOKENS = 32
DOCUMENT_TOKENS_PER_STREAM = DOCUMENT_BLOCKS * DOCUMENT_BLOCK_TOKENS
DOCUMENT_HISTORY_BLOCKS = 16
DOCUMENT_DEPENDENCY_LAGS = (1, 4, 8, 16)
DOCUMENT_SECTION_ROLES = (
    "historical evidence",
    "risk analysis",
    "practical recommendations",
)
DOCUMENT_CODEWORDS = (
    "amber", "anchor", "basalt", "beacon", "bramble", "cactus", "canyon", "cedar",
    "cinder", "cobalt", "copper", "coral", "dahlia", "delta", "dune", "ember",
    "fathom", "fennel", "flint", "forge", "gable", "garnet", "geyser", "glacier",
    "granite", "harbor", "hazel", "indigo", "ivory", "jasper", "juniper", "kelp",
    "lantern", "larch", "lichen", "marble", "meadow", "mesa", "nectar", "nimbus",
    "onyx", "opal", "orchid", "pewter", "pumice", "quarry", "quartz", "ravine",
    "saffron", "sable", "shale", "sienna", "slate", "spruce", "summit", "talon",
    "thicket", "topaz", "tundra", "umber", "verbena", "willow", "zenith", "zephyr",
)
if len(DOCUMENT_CODEWORDS) != 64 or len(set(DOCUMENT_CODEWORDS)) != 64:
    raise RuntimeError("The document codeword alphabet must contain 64 unique entries.")

# Every source block is used once, every target block is unique, and each lag
# appears four times. The remaining sixteen blocks carry local prose and form
# a surface-matched nondependency control inside the same document.
DOCUMENT_DEPENDENCY_SCHEDULE = {
    1: (0, 1),
    5: (4, 1),
    9: (8, 1),
    13: (12, 1),
    6: (2, 4),
    10: (6, 4),
    14: (10, 4),
    18: (14, 4),
    11: (3, 8),
    15: (7, 8),
    19: (11, 8),
    23: (15, 8),
    17: (1, 16),
    21: (5, 16),
    25: (9, 16),
    29: (13, 16),
}


def validate_document_schedule() -> None:
    """Prove the registered schedule is causal, unique, and horizon-complete."""

    targets = tuple(DOCUMENT_DEPENDENCY_SCHEDULE)
    sources = tuple(source for source, _ in DOCUMENT_DEPENDENCY_SCHEDULE.values())
    lags = tuple(lag for _, lag in DOCUMENT_DEPENDENCY_SCHEDULE.values())
    if len(set(targets)) != len(targets) or len(set(sources)) != len(sources):
        raise RuntimeError("Document dependency targets and sources must each be unique.")
    for target, (source, lag) in DOCUMENT_DEPENDENCY_SCHEDULE.items():
        if target - source != lag or not 0 <= source < target < DOCUMENT_BLOCKS:
            raise RuntimeError("Document dependency schedule contains a noncausal edge.")
    if set(lags) != set(DOCUMENT_DEPENDENCY_LAGS):
        raise RuntimeError("Document dependency schedule does not cover every registered lag.")
    if any(lags.count(lag) != 4 for lag in DOCUMENT_DEPENDENCY_LAGS):
        raise RuntimeError("Every document dependency lag must occur exactly four times.")
    if max(lags) != DOCUMENT_HISTORY_BLOCKS:
        raise RuntimeError("Document dependency horizon and bus history must match.")


validate_document_schedule()


__all__ = [
    "DOCUMENT_BLOCKS",
    "DOCUMENT_BLOCK_TOKENS",
    "DOCUMENT_CODEWORDS",
    "DOCUMENT_CONTRACT_VERSION",
    "DOCUMENT_DEPENDENCY_LAGS",
    "DOCUMENT_DEPENDENCY_SCHEDULE",
    "DOCUMENT_HISTORY_BLOCKS",
    "DOCUMENT_SECTION_ROLES",
    "DOCUMENT_TOKENS_PER_STREAM",
    "validate_document_schedule",
]
