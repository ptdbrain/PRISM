from pathlib import Path

from prism.cli.demo import run_demo


def test_prism_demo_runs_end_to_end(tmp_path: Path) -> None:
    result = run_demo(output_root=tmp_path)

    assert (tmp_path / "checkpoint" / "model.pt").exists()
    assert (tmp_path / "profile.json").exists()
    assert (tmp_path / "quic_assignment.json").exists()
    assert (tmp_path / "rtn" / "manifest.pt").exists()
    assert result["runtime_summary"]["backend_by_layer"]
