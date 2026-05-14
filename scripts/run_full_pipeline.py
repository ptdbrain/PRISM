"""Cross-platform PRISM full-pipeline runner.

This script is intentionally stdlib-only so the Windows ``.bat`` and shell
wrappers can use it before importing PRISM or torch.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _split_csv(values: str | list[str] | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = values
    items: list[str] = []
    for value in raw_values:
        items.extend(part.strip() for part in value.split(","))
    return [item for item in items if item]


def _sanitize_model_name(model_name: str) -> str:
    sanitized = model_name
    for char in ("/", ":", "\\", " "):
        sanitized = sanitized.replace(char, "_")
    return sanitized


def _display_cmd(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return shlex.join(cmd)


class PipelineRunner:
    def __init__(self, args: argparse.Namespace, repo_root: Path) -> None:
        self.args = args
        self.repo_root = repo_root
        self.out_root = Path(args.out_root)
        sanitized_model = _sanitize_model_name(args.model)
        self.stage0_dir = Path(args.stage0_dir or self.out_root / f"{sanitized_model}_stage0")
        self.run_dir = Path(args.run_dir or self.out_root / f"{sanitized_model}_full")
        self.profile_path = self.run_dir / "profile.json"
        self.assignment_path = self.run_dir / f"assignment_{args.budget}.json"
        self.quic_path = self.run_dir / f"quic_assignment_{args.budget}.json"
        self.rtn_dir = self.run_dir / "rtn"
        self.runtime_summary_path = self.run_dir / "runtime_summary.json"
        self.summary_json_path = self.run_dir / "summary_stats.json"
        self.summary_md_path = self.run_dir / "summary_stats.md"
        self.mlp_path = Path(args.mlp_path) if args.mlp_path else self.stage0_dir / "prism_mlp.pt"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = self.out_root / "logs"
        self.log_path = self.log_dir / f"{timestamp}_{sanitized_model}_full_pipeline.txt"

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    def stage_start(self, number: str, message: str) -> None:
        self.log()
        self.log(f"==== Stage {number} START: {message} ====")

    def stage_done(self, number: str, message: str) -> None:
        self.log(f"==== Stage {number} DONE: {message} ====")

    def stage_skip(self, number: str, message: str) -> None:
        self.log(f"==== Stage {number} SKIP: {message} ====")

    def expect_file(self, path: Path) -> None:
        if self.args.dry_run:
            return
        if not path.is_file():
            raise FileNotFoundError(f"Expected file was not created: {path}")

    def expect_dir(self, path: Path) -> None:
        if self.args.dry_run:
            return
        if not path.is_dir():
            raise FileNotFoundError(f"Expected directory was not created: {path}")

    def run_cmd(self, cmd: list[str]) -> None:
        self.log()
        self.log("+ " + _display_cmd(cmd))
        if self.args.dry_run:
            return

        process = subprocess.Popen(
            cmd,
            cwd=self.repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        with self.log_path.open("a", encoding="utf-8") as handle:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

    def python_cmd(self, *args: str) -> list[str]:
        return [sys.executable, *args]

    def trust_flag(self) -> list[str]:
        return ["--trust-remote-code"] if self.args.trust_remote_code else []

    def preflight_environment(self) -> None:
        if self.args.dry_run or self.args.skip_env_check:
            return

        self.log()
        self.log("==== Preflight START: Python and torch environment ====")
        probe = (
            "import sys\n"
            "print('Python executable:', sys.executable)\n"
            "print('Python version:', sys.version.split()[0])\n"
            "try:\n"
            "    import torch\n"
            "except Exception as exc:\n"
            "    raise SystemExit(f'Cannot import torch: {type(exc).__name__}: {exc}')\n"
            "print('Torch version:', torch.__version__)\n"
            "print('Torch CUDA build:', torch.version.cuda)\n"
            "print('CUDA available:', torch.cuda.is_available())\n"
            "print('CUDA device count:', torch.cuda.device_count())\n"
            "if sys.argv[1].startswith('cuda'):\n"
            "    if not torch.cuda.is_available():\n"
            "        raise SystemExit('CUDA was requested but this Python environment has CPU-only torch or cannot access the NVIDIA driver.')\n"
            "    if ':' in sys.argv[1]:\n"
            "        index = int(sys.argv[1].split(':', 1)[1])\n"
            "        if index >= torch.cuda.device_count():\n"
            "            raise SystemExit(f'CUDA device index {index} is out of range for {torch.cuda.device_count()} visible device(s).')\n"
        )
        self.run_cmd(self.python_cmd("-c", probe, self.args.device))
        self.log("==== Preflight DONE: environment is compatible with requested device ====")

    def run(self) -> None:
        os.chdir(self.repo_root)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if not self.args.dry_run:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        self.log("PRISM full pipeline")
        self.log(f"repo: {self.repo_root}")
        self.log(f"target model: {self.args.model}")
        self.log(f"run dir: {self.run_dir}")
        self.log(f"log file: {self.log_path}")
        if self.args.dry_run:
            self.log("mode: dry-run (commands are printed, not executed)")

        self.preflight_environment()
        self.run_stage0()
        self.run_stage1()
        self.run_stage2()
        final_assignment = self.run_quic()
        self.run_rtn()
        self.run_stage4(final_assignment)
        self.run_summary(final_assignment)

        self.log()
        self.log("Done.")
        self.log(f"Profile: {self.profile_path}")
        self.log(f"Assignment: {self.assignment_path}")
        self.log(f"Final assignment: {final_assignment}")
        self.log(f"RTN artifacts: {self.rtn_dir}")
        self.log(f"Runtime summary: {self.runtime_summary_path}")
        self.log(f"Summary JSON: {self.summary_json_path}")
        self.log(f"Summary report: {self.summary_md_path}")
        self.log(f"Log file: {self.log_path}")

    def run_stage0(self) -> None:
        if not self.args.run_stage0:
            self.stage_skip("0", f"Using existing MLP at {self.mlp_path}")
            self.expect_file(self.mlp_path)
            return

        self.stage_start("0", "Build sensitivity dataset and train meta-learner")
        if self.args.stage0_use_shards:
            self.run_cmd(
                self.python_cmd(
                    "scripts/stage0_sharded.py",
                    "--models",
                    self.args.stage0_models,
                    "--output-dir",
                    str(self.stage0_dir),
                    "--num-shards",
                    str(self.args.num_shards),
                    "--start-shard",
                    str(self.args.start_shard),
                    "--end-shard",
                    str(self.args.end_shard),
                    "--group-size",
                    str(self.args.group_size),
                    "--epochs",
                    str(self.args.epochs),
                )
            )
        else:
            self.run_cmd(
                self.python_cmd(
                    "-m",
                    "prism.cli.train_meta",
                    "--output-dir",
                    str(self.stage0_dir),
                    "--epochs",
                    str(self.args.epochs),
                    "--group-size",
                    str(self.args.group_size),
                    "--model-names",
                    *_split_csv(self.args.stage0_models),
                )
            )
        self.mlp_path = self.stage0_dir / "prism_mlp.pt"
        self.expect_file(self.mlp_path)
        self.stage_done("0", f"Meta-learner ready at {self.mlp_path}")

    def run_stage1(self) -> None:
        if not self.args.run_stage1:
            self.stage_skip("1", f"Reusing profile at {self.profile_path}")
            self.expect_file(self.profile_path)
            return
        self.stage_start("1", "Profile model layers and predict sensitivity")
        self.run_cmd(
            self.python_cmd(
                "-m",
                "prism.cli.profile",
                "--model-id-or-path",
                self.args.model,
                "--family",
                self.args.family,
                "--mlp-path",
                str(self.mlp_path),
                "--group-size",
                str(self.args.group_size),
                "--hidden-size",
                str(self.args.hidden_size),
                "--num-layers",
                str(self.args.num_layers),
                "--device",
                self.args.device,
                "--output-path",
                str(self.profile_path),
                *self.trust_flag(),
            )
        )
        self.expect_file(self.profile_path)
        self.stage_done("1", f"Profile written to {self.profile_path}")

    def run_stage2(self) -> None:
        if not self.args.run_stage2:
            self.stage_skip("2", f"Reusing assignment at {self.assignment_path}")
            self.expect_file(self.assignment_path)
            return
        self.stage_start("2", "Solve mixed-bit assignments")
        self.run_cmd(
            self.python_cmd(
                "-m",
                "prism.cli.assign",
                "--profile-path",
                str(self.profile_path),
                "--budget",
                str(self.args.budget),
                "--output-path",
                str(self.assignment_path),
            )
        )
        self.expect_file(self.assignment_path)

        for budget in _split_csv(self.args.budgets):
            if budget == str(self.args.budget):
                continue
            sweep_path = self.run_dir / f"assignment_{budget}.json"
            self.run_cmd(
                self.python_cmd(
                    "-m",
                    "prism.cli.assign",
                    "--profile-path",
                    str(self.profile_path),
                    "--budget",
                    budget,
                    "--output-path",
                    str(sweep_path),
                )
            )
            self.expect_file(sweep_path)
        self.stage_done("2", f"Assignments written under {self.run_dir}")

    def run_quic(self) -> Path:
        if not self.args.run_quic:
            self.stage_skip("2.5", "Using assignment without QUIC")
            return self.assignment_path
        self.stage_start("2.5", "Run synthetic-only QUIC correction")
        self.run_cmd(
            self.python_cmd(
                "-m",
                "prism.cli.quic",
                "--model-id-or-path",
                self.args.model,
                "--family",
                self.args.family,
                "--device",
                self.args.device,
                "--profile-path",
                str(self.profile_path),
                "--assignment-path",
                str(self.assignment_path),
                "--output-path",
                str(self.quic_path),
                "--hidden-size",
                str(self.args.hidden_size),
                "--seq-len",
                str(self.args.seq_len),
                *self.trust_flag(),
            )
        )
        self.expect_file(self.quic_path)
        self.stage_done("2.5", f"QUIC assignment written to {self.quic_path}")
        return self.quic_path

    def run_rtn(self) -> None:
        if not self.args.run_rtn:
            self.stage_skip("3", f"Reusing RTN artifacts at {self.rtn_dir}")
            self.expect_file(self.rtn_dir / "manifest.json")
            return
        self.stage_start("3", "Precompute RTN artifacts")
        self.run_cmd(
            self.python_cmd(
                "-m",
                "prism.cli.precompute_rtn",
                "--model-id-or-path",
                self.args.model,
                "--family",
                self.args.family,
                "--device",
                self.args.device,
                "--group-size",
                str(self.args.group_size),
                "--hidden-size",
                str(self.args.hidden_size),
                "--num-layers",
                str(self.args.num_layers),
                "--output-dir",
                str(self.rtn_dir),
                *self.trust_flag(),
            )
        )
        self.expect_file(self.rtn_dir / "manifest.json")
        self.expect_dir(self.rtn_dir / "layers")
        self.stage_done("3", f"RTN artifacts written to {self.rtn_dir}")

    def run_stage4(self, final_assignment: Path) -> None:
        if not self.args.run_stage4:
            self.stage_skip("4", "Runtime assembly skipped")
            return
        self.stage_start("4", "Assemble runtime model")
        cmd = self.python_cmd(
            "-m",
            "prism.cli.run",
            "--model-id-or-path",
            self.args.model,
            "--family",
            self.args.family,
            "--device",
            self.args.device,
            "--artifact-root",
            str(self.rtn_dir),
            "--assignment-path",
            str(final_assignment),
            "--hidden-size",
            str(self.args.hidden_size),
            "--num-layers",
            str(self.args.num_layers),
            "--prompt",
            self.args.prompt,
            "--max-new-tokens",
            str(self.args.max_new_tokens),
            "--summary-path",
            str(self.runtime_summary_path),
        )
        if self.args.execute:
            cmd.append("--execute")
        cmd.extend(self.trust_flag())
        self.run_cmd(cmd)
        self.expect_file(self.runtime_summary_path)
        self.stage_done("4", "Runtime assembly completed")

    def run_summary(self, final_assignment: Path) -> None:
        if not self.args.run_summary:
            self.stage_skip("5", "Run summary skipped")
            return
        self.stage_start("5", "Summarize artifacts and effectiveness metadata")
        cmd = self.python_cmd(
            "scripts/summarize_pipeline_run.py",
            "--run-dir",
            str(self.run_dir),
            "--assignment-path",
            str(final_assignment),
            "--output-json",
            str(self.summary_json_path),
            "--output-md",
            str(self.summary_md_path),
            "--quiet",
        )
        eval_results = _split_csv(self.args.eval_results)
        if eval_results:
            cmd.extend(["--eval-results", *eval_results])
        self.run_cmd(cmd)
        self.expect_file(self.summary_json_path)
        self.expect_file(self.summary_md_path)
        self.stage_done("5", f"Summary report written to {self.summary_md_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full PRISM pipeline.")
    parser.add_argument("--model", default=_env("MODEL_NAME", "Qwen/Qwen2.5-1.5B"))
    parser.add_argument("--stage0-models", default=_env("STAGE0_MODELS", _env("MODEL_NAME", "Qwen/Qwen2.5-1.5B")))
    parser.add_argument("--mlp-path", default=os.environ.get("MLP_PATH"))
    parser.add_argument("--out-root", default=_env("OUT_ROOT", "artifacts/prism"))
    parser.add_argument("--stage0-dir", default=os.environ.get("STAGE0_DIR"))
    parser.add_argument("--run-dir", default=os.environ.get("RUN_DIR"))
    parser.add_argument("--run-stage0", action="store_true", default=_env_bool("RUN_STAGE0"))
    parser.add_argument("--stage0-use-shards", action="store_true", default=_env_bool("STAGE0_USE_SHARDS"))
    parser.add_argument("--num-shards", type=int, default=int(_env("NUM_SHARDS", "16")))
    parser.add_argument("--start-shard", type=int, default=int(_env("START_SHARD", "0")))
    parser.add_argument("--end-shard", type=int, default=int(_env("END_SHARD", _env("NUM_SHARDS", "16"))))
    parser.add_argument("--epochs", type=int, default=int(_env("EPOCHS", "100")))
    parser.add_argument("--group-size", type=int, default=int(_env("GROUP_SIZE", "128")))
    parser.add_argument("--hidden-size", type=int, default=int(_env("HIDDEN_SIZE", "16")))
    parser.add_argument("--num-layers", type=int, default=int(_env("NUM_LAYERS", "4")))
    parser.add_argument("--seq-len", type=int, default=int(_env("SEQ_LEN", "8")))
    parser.add_argument("--device", default=_env("DEVICE", "cuda"))
    parser.add_argument("--family", default=_env("FAMILY", "auto"))
    parser.add_argument("--budget", default=_env("BUDGET", "3.0"))
    parser.add_argument("--budgets", nargs="*", default=_env("BUDGETS", "2.5,2.75,3.0,3.25,3.5"))
    parser.add_argument("--skip-stage1", action="store_false", dest="run_stage1", default=_env_bool("RUN_STAGE1", True))
    parser.add_argument("--skip-stage2", action="store_false", dest="run_stage2", default=_env_bool("RUN_STAGE2", True))
    parser.add_argument("--skip-quic", action="store_false", dest="run_quic", default=_env_bool("RUN_QUIC", True))
    parser.add_argument("--skip-rtn", action="store_false", dest="run_rtn", default=_env_bool("RUN_RTN", True))
    parser.add_argument("--skip-run", action="store_false", dest="run_stage4", default=_env_bool("RUN_STAGE4", True))
    parser.add_argument("--skip-summary", action="store_false", dest="run_summary", default=_env_bool("RUN_SUMMARY", True))
    parser.add_argument("--eval-results", nargs="*", default=os.environ.get("RESEARCH_EVAL_RESULTS"))
    parser.add_argument("--execute", action="store_true", default=_env_bool("PRISM_EXECUTE"))
    parser.add_argument("--trust-remote-code", action="store_true", default=_env_bool("TRUST_REMOTE_CODE"))
    parser.add_argument("--prompt", default=_env("PRISM_PROMPT", "Hello"))
    parser.add_argument("--max-new-tokens", type=int, default=int(_env("MAX_NEW_TOKENS", "16")))
    parser.add_argument("--dry-run", action="store_true", default=_env_bool("DRY_RUN"))
    parser.add_argument("--skip-env-check", action="store_true", default=_env_bool("SKIP_ENV_CHECK"))
    return parser


def main(argv: list[str] | None = None) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    args = build_parser().parse_args(argv)
    PipelineRunner(args, repo_root).run()


if __name__ == "__main__":
    main()
