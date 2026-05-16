"""PRISM-Atlas action-aware quantization policy components."""

__all__ = [
    "QuantizationAction",
    "action_from_id",
    "build_action_space",
    "build_atlas_profile",
    "solve_atlas_assignment",
]


def __getattr__(name: str):
    if name in {"QuantizationAction", "action_from_id", "build_action_space"}:
        from prism.atlas import actions

        return getattr(actions, name)
    if name == "build_atlas_profile":
        from prism.atlas.profile import build_atlas_profile

        return build_atlas_profile
    if name == "solve_atlas_assignment":
        from prism.atlas.solver import solve_atlas_assignment

        return solve_atlas_assignment
    raise AttributeError(f"module 'prism.atlas' has no attribute {name!r}")
