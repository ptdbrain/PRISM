from prism.atlas.actions import QuantizationAction, action_feature_names, build_action_space


def test_default_action_space_is_stable() -> None:
    actions = build_action_space()

    assert len(actions) == 12
    assert actions[0].action_id == "rtn_b2_g64_none"
    assert actions[-1].action_id == "rtn_b4_g128_hadamard"


def test_action_memory_and_features_are_explicit() -> None:
    action = QuantizationAction(bits=3, group_size=64, transform="hadamard", backend="rtn")

    features = action.to_feature_dict(transform_supported=True)

    assert set(features) == set(action_feature_names())
    assert features["bits"] == 3.0
    assert features["group_size"] == 64.0
    assert features["transform_hadamard"] == 1.0
    assert action.memory_cost_bits([8, 8]) > 0
