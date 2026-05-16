from prism.atlas.actions import QuantizationAction
from prism.atlas.profile import build_atlas_profile
from prism.atlas.solver import solve_atlas_assignment
from prism.models.mock_transformer import MockTransformerLM


def test_build_atlas_profile_records_response_surface() -> None:
    model = MockTransformerLM(hidden_size=8, num_layers=1)
    actions = [
        QuantizationAction(bits=2, group_size=64, transform="none"),
        QuantizationAction(bits=4, group_size=64, transform="hadamard"),
    ]

    profile = build_atlas_profile(model, actions=actions, model_id="mock")

    assert profile["method"] == "prism-atlas-v1"
    assert profile["layers"][0]["responses"][actions[0].action_id]["memory_cost_bits"] > 0


def test_risk_lambda_can_change_assignment_choice() -> None:
    profile = {
        "method": "prism-atlas-v1",
        "layers": [
            {
                "layer_name": "a",
                "num_params": 16,
                "responses": {
                    "rtn_b2_g64_none": {
                        "action": {"bits": 2, "group_size": 64, "transform": "none", "backend": "rtn"},
                        "mean_damage": 0.1,
                        "uncertainty": 10.0,
                        "log_variance": 4.0,
                        "ranking_score": -0.1,
                        "memory_cost_bits": 32.0,
                    },
                    "rtn_b4_g64_none": {
                        "action": {"bits": 4, "group_size": 64, "transform": "none", "backend": "rtn"},
                        "mean_damage": 0.2,
                        "uncertainty": 0.0,
                        "log_variance": -20.0,
                        "ranking_score": -0.2,
                        "memory_cost_bits": 64.0,
                    },
                },
            }
        ],
    }

    mean_only = solve_atlas_assignment(profile, budget_bits=64.0, objective_mode="mean", risk_lambda=0.0)
    risk = solve_atlas_assignment(profile, budget_bits=64.0, objective_mode="risk", risk_lambda=1.0)

    assert mean_only["actions"]["a"] == "rtn_b2_g64_none"
    assert risk["actions"]["a"] == "rtn_b4_g64_none"
    assert risk["bits"]["a"] == 4
