from prism.assign.optimize import assign_bits
from prism.assignment.lp_solver import solve_lp
from prism.data.schemas import ProfileArtifact, ProfileLayerRecord


def test_assign_bits_respects_budget_and_fixed_layers() -> None:
    artifact = ProfileArtifact(
        model_id="mock-transformer",
        model_family="synthetic",
        layers=[
            ProfileLayerRecord(
                layer_name="layers.0.self_attn.o_proj",
                module_type="o_proj",
                shape=[8, 8],
                num_params=64,
                features={},
                raw_score=2.0,
                adjusted_score=2.0,
                fixed_4bit=True,
                fixed_reason="o_proj_rule",
                memory_cost_by_bit={"2": 128.0, "3": 192.0, "4": 256.0},
            ),
            ProfileLayerRecord(
                layer_name="layers.0.self_attn.v_proj",
                module_type="v_proj",
                shape=[8, 8],
                num_params=64,
                features={},
                raw_score=3.0,
                adjusted_score=4.5,
                fixed_4bit=False,
                fixed_reason="",
                memory_cost_by_bit={"2": 128.0, "3": 192.0, "4": 256.0},
            ),
            ProfileLayerRecord(
                layer_name="layers.0.mlp.down_proj",
                module_type="down_proj",
                shape=[8, 8],
                num_params=64,
                features={},
                raw_score=1.0,
                adjusted_score=1.0,
                fixed_4bit=False,
                fixed_reason="",
                memory_cost_by_bit={"2": 128.0, "3": 192.0, "4": 256.0},
            ),
        ],
        metadata={},
    )

    result = assign_bits(artifact, target_average_bits=3.0)

    assert result["bits"]["layers.0.self_attn.o_proj"] == 4
    assert result["average_bits"] <= 3.0
    assert result["solver"]["exact"] is True
    assert result["memory_cost_bits"] <= result["budget_memory_bits"] + 1e-6


def test_solve_lp_uses_discrete_one_hot_costs() -> None:
    profile = {
        "a": {
            "shape": (4, 4),
            "num_params": 16,
            "sensitivity": {2: 10.0, 3: 5.0, 4: 0.0},
            "memory_cost_bits": {2: 32.0, 3: 48.0, 4: 64.0},
            "forced_bits": None,
        },
        "b": {
            "shape": (4, 4),
            "num_params": 16,
            "sensitivity": {2: 1.0, 3: 0.5, 4: 0.0},
            "memory_cost_bits": {2: 32.0, 3: 48.0, 4: 64.0},
            "forced_bits": None,
        },
    }
    budget = profile["a"]["memory_cost_bits"][4] + profile["b"]["memory_cost_bits"][2]
    bits, diagnostics = solve_lp(profile, budget, return_diagnostics=True)

    assert bits == {"a": 4, "b": 2}
    assert diagnostics["method"] in {"milp", "frontier-dp"}
