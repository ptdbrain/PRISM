from prism.atlas.actions import QuantizationAction
from prism.atlas.calibration import calibration_report
from prism.atlas.profile import build_atlas_profile
from prism.atlas.protocols import benchmark_protocol, cleanup_plan, transfer_protocol
from prism.atlas.scoring import analytic_action_response
from prism.atlas.solver import solve_atlas_assignment
from prism.models.mock_transformer import MockTransformerLM


def test_analytic_scorer_exposes_kappa_distortion_and_schema() -> None:
    model = MockTransformerLM(hidden_size=8, num_layers=1)
    action = QuantizationAction(bits=2, group_size=64, transform="none")
    layer_name, module = next(iter(model.named_modules()))
    del layer_name

    response = analytic_action_response(module.layers[0].self_attn.q_proj.weight, action, {"module_type": "q_proj"})

    assert response["scorer"] == "analytic"
    assert response["analytic_damage"] == response["mean_damage"]
    assert response["kappa"] > 0.0
    assert response["distortion"] > 0.0


def test_profile_v2_records_validity_damage_definition_and_scorer_mode() -> None:
    model = MockTransformerLM(hidden_size=8, num_layers=1)
    actions = [
        QuantizationAction(bits=2, group_size=64, transform="none"),
        QuantizationAction(bits=2, group_size=64, transform="hadamard"),
    ]

    profile = build_atlas_profile(model, actions=actions, model_id="mock", scorer="analytic")
    response = profile["layers"][0]["responses"][actions[0].action_id]
    hadamard_response = profile["layers"][0]["responses"][actions[1].action_id]

    assert profile["schema_version"] == "prism-atlas-profile-v2"
    assert profile["artifact_type"] == "ResponseSurfaceArtifactV2"
    assert profile["damage_definition"]["primary_label"] == "delta_nll_or_jsd_logits"
    assert response["valid_action"] is True
    assert response["runtime_supported"] is True
    assert response["materialization_supported"] is True
    assert response["latency_proxy"] > 0.0
    assert hadamard_response["runtime_supported"] is False
    assert hadamard_response["fallback_backend"] == "rtn"


def test_solver_filters_invalid_actions_by_default() -> None:
    profile = {
        "schema_version": "prism-atlas-profile-v2",
        "layers": [
            {
                "layer_name": "a",
                "num_params": 16,
                "responses": {
                    "bad": {
                        "action": {"bits": 2, "group_size": 64, "transform": "hadamard", "backend": "rtn"},
                        "mean_damage": 0.0,
                        "uncertainty": 0.0,
                        "memory_cost_bits": 32.0,
                        "valid_action": False,
                    },
                    "good": {
                        "action": {"bits": 4, "group_size": 64, "transform": "none", "backend": "rtn"},
                        "mean_damage": 1.0,
                        "uncertainty": 0.0,
                        "memory_cost_bits": 64.0,
                        "valid_action": True,
                    },
                },
            }
        ],
    }

    result = solve_atlas_assignment(profile, budget_bits=64.0, objective_mode="mean")

    assert result["schema_version"] == "prism-atlas-assignment-v2"
    assert result["actions"]["a"] == "good"
    assert result["invalid_actions_filtered"] == 1


def test_calibration_report_contains_required_uncertainty_metrics() -> None:
    report = calibration_report(
        predicted_mean=[0.1, 0.2, 0.8, 1.0],
        predicted_uncertainty=[0.05, 0.10, 0.40, 0.50],
        measured_damage=[0.12, 0.18, 0.75, 1.2],
        top_fraction=0.5,
    )

    assert set(report) >= {"nll", "spearman", "risk_ece", "top_k_risky_recall"}
    assert report["top_k_risky_recall"] == 1.0


def test_benchmark_transfer_and_cleanup_protocols_are_explicit() -> None:
    benchmark = benchmark_protocol()
    transfer = transfer_protocol()
    cleanup = cleanup_plan()

    assert "WikiText-2 PPL" in benchmark["metrics"]
    assert "C4 PPL" in benchmark["metrics"]
    assert "PRISM-Atlas hybrid risk-aware" in benchmark["systems"]
    assert "Qwen2.5-1.5B" in transfer["target_unseen_models"]
    assert transfer["settings"] == ["zero-target-label", "light-target-calibration", "full-target-label-oracle"]
    assert "archive prism/meta/" in cleanup["phase_1"]
