"""Unified public API for PRISM real-model and demo execution."""

from __future__ import annotations

import logging
from pathlib import Path

from prism.assign.optimize import assign_bits
from prism.data.io import save_json
from prism.data.synthetic import make_sensitivity_dataset
from prism.meta.train import train_meta_learner
from prism.profile.pipeline import profile_model
from prism.quic.pipeline import run_quic_correction
from prism.rtn.precompute import precompute_model_rtn
from prism.runtime.assemble import assemble_runtime_model
from prism.support.model_loading import load_model_bundle

logger = logging.getLogger(__name__)


class PRISM:
    def __init__(
        self,
        model_id: str | None = None,
        *,
        model=None,
        family: str | None = None,
        artifact_dir: str | Path = "artifacts/prism",
        device: str | None = None,
        torch_dtype: str | None = "float16",
        trust_remote_code: bool = False,
        dataset: str | None = None,
        hidden_size: int = 16,
        num_layers: int = 4,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.dataset = dataset
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.bundle = load_model_bundle(
            model_id_or_path=model_id,
            model=model,
            family=family,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.last_run: dict[str, object] | None = None

    def _ensure_demo_checkpoint(self) -> Path:
        checkpoint_dir = self.artifact_dir / "checkpoint"
        model_path = checkpoint_dir / "model.pt"
        if model_path.exists():
            return checkpoint_dir

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        train_meta_learner(
            make_sensitivity_dataset(self.bundle.model),
            output_dir=checkpoint_dir,
            model=self.bundle.model,
            epochs=5,
            seed=0,
        )
        return checkpoint_dir

    def run(
        self,
        *,
        target_bits: float = 3.0,
        group_size: int = 128,
        mlp_path: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        quic_rounds: int = 2,
    ):
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        resolved_mlp = Path(mlp_path) if mlp_path is not None else None
        resolved_checkpoint = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if resolved_mlp is None and resolved_checkpoint is None:
            if self.bundle.is_demo:
                resolved_checkpoint = self._ensure_demo_checkpoint()
            else:
                raise ValueError("Real-model PRISM run requires `mlp_path` or `checkpoint_dir`.")

        profile_path = self.artifact_dir / "profile.json"
        assignment_path = self.artifact_dir / "assignment.json"
        quic_path = self.artifact_dir / "quic_assignment.json"
        rtn_dir = self.artifact_dir / "rtn"

        try:
            profile = profile_model(
                model=self.bundle.model,
                checkpoint_dir=resolved_checkpoint,
                mlp_path=resolved_mlp,
                output_path=profile_path,
                model_id=self.bundle.model_id,
                model_family=self.bundle.model_family,
                group_size=group_size,
            )
        except Exception as exc:
            raise RuntimeError(f"Stage 1 (Profiling) failed: {exc}") from exc

        try:
            assignment = assign_bits(profile, target_average_bits=target_bits)
            save_json(assignment_path, assignment)
        except ValueError as exc:
            raise RuntimeError(f"Stage 2 (Assignment) failed, likely infeasible budget: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Stage 2 (Assignment) failed: {exc}") from exc

        try:
            quic_assignment = run_quic_correction(
                model=self.bundle.model,
                profile_artifact=profile,
                assignment=assignment,
                hidden_size=self.bundle.hidden_size or 8,
                seq_len=4,
                rounds=quic_rounds,
                group_size=group_size,
            )
            save_json(quic_path, quic_assignment)
        except Exception as exc:
            logger.warning("Stage 2.5 (QUIC) failed; using pre-QUIC assignment: %s", exc)
            quic_assignment = assignment

        try:
            manifest = precompute_model_rtn(
                model=self.bundle.model,
                output_dir=rtn_dir,
                group_size=group_size,
                model_id=self.bundle.model_id,
            )
        except Exception as exc:
            raise RuntimeError(f"Stage 3 (RTN precomputation) failed: {exc}") from exc

        try:
            runtime_model, runtime_summary = assemble_runtime_model(
                base_model=self.bundle.model,
                manifest=manifest,
                assignment=quic_assignment,
                artifact_root=rtn_dir,
            )
        except Exception as exc:
            raise RuntimeError(f"Stage 4 (Runtime assembly) failed: {exc}") from exc
        setattr(runtime_model, "backend_summary", runtime_summary["backend_by_layer"])

        self.last_run = {
            "profile": profile,
            "assignment": assignment,
            "quic_assignment": quic_assignment,
            "manifest": manifest,
            "runtime_summary": runtime_summary,
            "runtime_model": runtime_model,
        }
        return runtime_model
