from prism.atlas.quic import select_uncertain_layers


def test_select_uncertain_layers_uses_chosen_action_uncertainty() -> None:
    profile = {
        "layers": [
            {"layer_name": "a", "responses": {"x": {"uncertainty": 0.1}}},
            {"layer_name": "b", "responses": {"x": {"uncertainty": 3.0}}},
            {"layer_name": "c", "responses": {"x": {"uncertainty": 1.0}}},
        ]
    }
    assignment = {"actions": {"a": "x", "b": "x", "c": "x"}}

    assert select_uncertain_layers(profile, assignment, top_fraction=0.34) == ["b"]
