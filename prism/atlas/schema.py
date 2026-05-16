"""Versioned PRISM-Atlas artifact schema constants."""

PROFILE_SCHEMA_VERSION = "prism-atlas-profile-v2"
ASSIGNMENT_SCHEMA_VERSION = "prism-atlas-assignment-v2"
PROFILE_ARTIFACT_TYPE = "ResponseSurfaceArtifactV2"
ASSIGNMENT_ARTIFACT_TYPE = "AssignmentV2"

DAMAGE_DEFINITION = {
    "primary_label": "delta_nll_or_jsd_logits",
    "secondary_validation": ["WikiText-2 PPL", "C4 PPL"],
    "final_metric": "zero_shot_average",
    "low_resource_surrogate": "rtn_relative_mse_with_kappa_weighting",
    "notes": (
        "Atlas v1 can bootstrap from RTN distortion when end-task labels are unavailable, "
        "but paper claims should validate against NLL/JSD and downstream tasks."
    ),
}
