# PRISM-Atlas v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PRISM-Atlas v1 with action-aware response surfaces, risk-aware assignment, and uncertainty-guided correction planning while keeping existing PRISM-original workflows compatible.

**Architecture:** Implement Atlas as a new `prism.atlas` package plus new CLI entrypoints. Existing Stage 0-4 modules remain baseline-compatible, while Atlas artifacts include both selected action IDs and legacy `bits` maps for current runtime assembly.

**Tech Stack:** Python 3.10+, PyTorch, SciPy optional, current PRISM memory accounting and model loading helpers.

---

### Task 1: Atlas Action Schema

**Files:**
- Create: `prism/atlas/__init__.py`
- Create: `prism/atlas/actions.py`
- Test: `tests/prism/test_atlas_actions.py`

- [ ] **Step 1: Write failing tests**

```python
from prism.atlas.actions import QuantizationAction, build_action_space, action_feature_names


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/prism/test_atlas_actions.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'prism.atlas'`.

- [ ] **Step 3: Implement action dataclass and action space**

Create `QuantizationAction`, `build_action_space`, `action_from_id`, and `action_feature_names`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/prism/test_atlas_actions.py -q`
Expected: PASS.

### Task 2: Response Surface Model And Dataset

**Files:**
- Create: `prism/atlas/transforms.py`
- Create: `prism/atlas/response.py`
- Create: `prism/atlas/dataset.py`
- Test: `tests/prism/test_atlas_response.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

import torch

from prism.atlas.actions import build_action_space
from prism.atlas.dataset import build_response_dataset
from prism.atlas.response import ResponseSurfaceMLP, train_response_surface, load_response_surface
from prism.models.mock_transformer import MockTransformerLM


def test_response_model_outputs_mean_variance_and_ranking() -> None:
    model = ResponseSurfaceMLP(layer_feature_dim=4, action_feature_dim=3)
    mean, log_var, ranking = model(torch.zeros(2, 4), torch.zeros(2, 3))
    assert mean.shape == (2,)
    assert log_var.shape == (2,)
    assert ranking.shape == (2,)
    assert torch.isfinite(mean).all()


def test_response_dataset_and_training_checkpoint(tmp_path: Path) -> None:
    mock = MockTransformerLM(hidden_size=8, num_layers=1)
    actions = build_action_space(bits=(2, 4), group_sizes=(64,), transforms=("none", "hadamard"))
    dataset_path = tmp_path / "atlas_data.pt"
    bundle = build_response_dataset([("mock", mock)], actions=actions, save_path=dataset_path)
    assert bundle["X_layer"].shape[0] == len(bundle["meta"])
    assert bundle["Y_mean"].shape[0] == len(bundle["meta"])
    ckpt = tmp_path / "atlas.pt"
    train_response_surface(dataset_path=str(dataset_path), epochs=2, save_path=str(ckpt))
    loaded = load_response_surface(str(ckpt))
    assert loaded.layer_feature_order
    assert loaded.action_feature_order
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/prism/test_atlas_response.py -q`
Expected: FAIL because `prism.atlas.response` is missing.

- [ ] **Step 3: Implement transform helpers, response MLP, dataset builder, trainer, loader**

Use deterministic RTN relative MSE as the v1 low-resource label source.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/prism/test_atlas_response.py -q`
Expected: PASS.

### Task 3: Atlas Profile And Risk-Aware Solver

**Files:**
- Create: `prism/atlas/profile.py`
- Create: `prism/atlas/solver.py`
- Test: `tests/prism/test_atlas_solver.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/prism/test_atlas_solver.py -q`
Expected: FAIL because profile/solver modules are missing.

- [ ] **Step 3: Implement Atlas profile builder and solver**

Use the existing realistic memory model for action costs and a Pareto-frontier dynamic program for multiple-choice assignment.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/prism/test_atlas_solver.py -q`
Expected: PASS.

### Task 4: Uncertainty-Guided QUIC Planning And CLIs

**Files:**
- Create: `prism/atlas/quic.py`
- Create: `prism/cli/atlas_train.py`
- Create: `prism/cli/atlas_profile.py`
- Create: `prism/cli/atlas_assign.py`
- Modify: `pyproject.toml`
- Modify: `tests/prism/test_cli_entrypoints.py`
- Test: `tests/prism/test_atlas_quic.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/prism/test_atlas_quic.py tests/prism/test_cli_entrypoints.py -q`
Expected: FAIL because `prism.atlas.quic` and CLI files are missing.

- [ ] **Step 3: Implement QUIC selector and CLI entrypoints**

`atlas-train` builds/trains response checkpoints, `atlas-profile` writes Atlas profiles, and `atlas-assign` writes action assignments plus legacy `bits` maps.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/prism/test_atlas_quic.py tests/prism/test_cli_entrypoints.py -q`
Expected: PASS.

### Task 5: Verification

**Files:**
- All files above

- [ ] **Step 1: Run focused Atlas tests**

Run: `pytest tests/prism/test_atlas_actions.py tests/prism/test_atlas_response.py tests/prism/test_atlas_solver.py tests/prism/test_atlas_quic.py tests/prism/test_cli_entrypoints.py -q`
Expected: PASS.

- [ ] **Step 2: Run existing affected tests**

Run: `pytest tests/prism/test_assignment.py tests/prism/test_profiling_spec.py tests/prism/test_profile_pipeline.py -q`
Expected: PASS.

- [ ] **Step 3: Run compile check**

Run: `python -m compileall prism tests -q`
Expected: exit code 0.
