#!/usr/bin/env python3
"""Run configured DRIFT CLI stages without duplicating the core algorithms."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid


STAGES = ("prepare", "rollout", "label", "extract", "score", "select")
ROLLOUT_OPTIONS = (
    "revision", "samples", "temperature", "top_p", "top_k", "max_new_tokens",
    "max_model_len", "seed",
)
EXTRACTION_OPTIONS = (
    "revision", "dimension", "projection_seed", "chunk_elements", "max_length",
    "dtype", "attention", "save_every", "exclude_parameters",
    "skip_no_supervision", "gradient_checkpointing",
)
SCORE_OPTIONS = ("fit_mode", "clip", "zscore")
BOOLEAN_OPTIONS = {"skip_no_supervision", "gradient_checkpointing", "allow_exclusions", "zscore"}
KNOWN_CONFIG_KEYS = set(ROLLOUT_OPTIONS + EXTRACTION_OPTIONS + SCORE_OPTIONS) | {
    "model", "backend", "fraction", "allow_exclusions",
}


@dataclass(frozen=True)
class Command:
    name: str
    argv: tuple[str, ...]
    output: Path
    gpu: str | None = None

    def record(self):
        return {"name": self.name, "argv": list(self.argv), "output": str(self.output),
                "environment": {"CUDA_VISIBLE_DEVICES": self.gpu} if self.gpu is not None else {}}

    def display(self):
        prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(self.gpu)} " if self.gpu is not None else ""
        return prefix + shlex.join(self.argv)


@dataclass(frozen=True)
class CommandGroup:
    name: str
    commands: tuple[Command, ...]


class PipelineFailure(RuntimeError):
    """A child failed; no later command group was started."""


def parse_gpus(value):
    parts = value.split(",")
    if not parts or any(not item.strip().isdigit() for item in parts):
        raise ValueError("--gpus must be a comma-separated list of nonnegative GPU indices")
    gpus = [str(int(item.strip())) for item in parts]
    if len(set(gpus)) != len(gpus):
        raise ValueError("--gpus must not contain duplicates")
    return gpus


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    unknown = set(config) - KNOWN_CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unsupported configuration keys: {sorted(unknown)}")
    for key in BOOLEAN_OPTIONS:
        if key in config and not isinstance(config[key], bool):
            raise ValueError(f"Configuration {key} must be boolean")
    return config


def _options(config, keys):
    arguments = []
    for key in keys:
        value = config.get(key)
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if key in BOOLEAN_OPTIONS:
            if value:
                arguments.append(flag)
        elif key == "clip":
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError("Configuration clip must be a two-element percentile list")
            arguments.extend([flag, *(str(item) for item in value)])
        else:
            arguments.extend([flag, str(value)])
    return arguments


def build_plan(args, config, *, python=None):
    """Build a stage plan using paths only; do not touch outputs or load a model."""
    gpus = parse_gpus(args.gpus)
    model = args.model if args.model is not None else config.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Supply a nonempty model in the config or with --model")
    backend = args.backend or config.get("backend", "vllm")
    if backend not in {"vllm", "transformers"}:
        raise ValueError("backend must be vllm or transformers")
    root = Path(args.output)
    prefix = [python or sys.executable, "-m", "drift"]
    corpus = root / "corpus.jsonl"
    anchors = root / "anchors.jsonl"
    rollout_dirs = [root / f"rollout-rank-{rank}" for rank in range(len(gpus))]
    corpus_dirs = [root / f"corpus-rank-{rank}" for rank in range(len(gpus))]
    anchor_dirs = [root / f"anchors-rank-{rank}" for rank in range(len(gpus))]

    def command(name, operation, output, arguments, gpu=None):
        argv = prefix + [operation] + [str(item) for item in arguments] + ["--output", str(output)]
        return Command(name, tuple(argv), Path(output), gpu)

    stages = {}
    stages["prepare"] = [CommandGroup("prepare", (
        command("prepare", "prepare", corpus, ["--input", args.corpus]),
    ))]
    stages["rollout"] = [CommandGroup("rollout", tuple(
        command(f"rollout-rank-{rank}", "rollout", destination,
                ["--queries", args.queries, "--model", model, "--backend", backend,
                 "--rank", rank, "--world-size", len(gpus), *_options(config, ROLLOUT_OPTIONS)], gpu)
        for rank, (gpu, destination) in enumerate(zip(gpus, rollout_dirs))
    ))]
    label_arguments = ["--queries", args.queries, "--rollouts",
                       *(directory / "rollouts.jsonl" for directory in rollout_dirs),
                       *_options(config, ("allow_exclusions",))]
    if args.external_labels:
        label_arguments += ["--external-labels", args.external_labels]
    stages["label"] = [CommandGroup("label", (
        command("label", "label", anchors, label_arguments),
    ))]
    stages["extract"] = []
    for name, input_path, directories, merged in [
        ("corpus", corpus, corpus_dirs, root / "corpus-features"),
        ("anchors", anchors, anchor_dirs, root / "anchor-features"),
    ]:
        stages["extract"].append(CommandGroup(f"extract-{name}", tuple(
            command(f"extract-{name}-rank-{rank}", "extract", destination,
                    ["--input", input_path, "--model", model, "--device", "cuda:0",
                     "--rank", rank, "--world-size", len(gpus), *_options(config, EXTRACTION_OPTIONS)], gpu)
            for rank, (gpu, destination) in enumerate(zip(gpus, directories))
        )))
        stages["extract"].append(CommandGroup(f"merge-{name}", (
            command(f"merge-{name}", "merge", merged, ["--shards", *directories]),
        )))
    stages["score"] = [CommandGroup("score", (
        command("score", "score", root / "scores",
                ["--corpus-features", root / "corpus-features", "--anchor-features", root / "anchor-features",
                 *_options(config, SCORE_OPTIONS)]),
    ))]
    stages["select"] = [CommandGroup("select", (
        command("select", "select", root / "selection",
                ["--scores-dir", root / "scores", "--corpus", corpus, *_options(config, ("fraction",))]),
    ))]
    chosen = STAGES if args.stage == "all" else (args.stage,)
    return [group for stage in chosen for group in stages[stage]]


def validate_all_verifiers(args):
    """Reject an all-stage launch that would need labels not yet supplied."""
    if args.stage != "all":
        return
    has_external = False
    with Path(args.queries).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("verifier"), dict):
                raise ValueError(f"Query line {line_number} needs a verifier object")
            has_external |= row["verifier"].get("type") == "external"
    if has_external and not args.external_labels:
        raise ValueError("--stage all includes external verifiers: run through rollout first, evaluate the responses, "
                         "then use --stage label with --external-labels, or supply labels for this run upfront")
    if args.external_labels and not Path(args.external_labels).is_file():
        raise ValueError("The supplied external-labels file does not exist")


def execute_plan(groups, output_root, *, metadata=None):
    """Advance only on success; abort this runner's process groups on failure."""
    for group in groups:
        for command in group.commands:
            if command.output.exists() or command.output.is_symlink():
                raise FileExistsError(f"Output already exists: {command.output}; use a fresh stage output")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logs = Path(output_root) / "runner_logs" / f"{timestamp}-{uuid.uuid4().hex[:8]}"
    logs.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "cwd": str(Path.cwd()), "metadata": metadata or {},
              "plan": [{"name": group.name, "commands": [command.record() for command in group.commands]}
                       for group in groups], "results": [], "logs": str(logs)}

    def save():
        report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = logs / "status.tmp.json"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, logs / "status.json")

    def stop_owned(active, reason):
        # Every entry was created below with start_new_session=True, so its PID
        # is the dedicated process-group ID. Never signal the caller's group,
        # a GPU device, or a process discovered outside this invocation.
        targets = [(process, entry) for process, _, entry in active if entry["status"] != "complete"]
        for process, entry in targets:
            if entry["status"] == "running":
                entry.update(status="aborted", reason=reason)
            try:
                os.killpg(process.pid, signal.SIGTERM)
                entry["term_requested"] = True
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 5.0
        for process, entry in targets:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        # A leader can exit while one of its worker processes remains. Probe
        # only the exact groups owned by this invocation before escalation.
        for process, entry in targets:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                pass
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    entry["kill_requested"] = True
                except ProcessLookupError:
                    pass  # The owned group exited between the probe and kill.
            try:
                entry["returncode"] = process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                entry["termination_pending"] = True
        save()

    save()
    try:
        for group in groups:
            report["active_group"] = group.name
            save()
            active = []
            failed = False
            try:
                for command in group.commands:
                    log_path = logs / f"{command.name}.log"
                    handle = log_path.open("x", encoding="utf-8")
                    handle.write(command.display() + "\n")
                    handle.flush()
                    entry = {"group": group.name, "name": command.name, "log": str(log_path),
                             "status": "starting", "returncode": None}
                    report["results"].append(entry)
                    environment = os.environ.copy()
                    if command.gpu is not None:
                        environment["CUDA_VISIBLE_DEVICES"] = command.gpu
                    try:
                        process = subprocess.Popen(command.argv, stdout=handle, stderr=subprocess.STDOUT,
                                                   env=environment, start_new_session=True)
                    except OSError as error:
                        handle.write(f"Launch failed: {error}\n")
                        handle.close()
                        entry.update(status="launch_failed", error=str(error))
                        failed = True
                        save()
                        break
                    entry.update(status="running", pid=process.pid)
                    active.append((process, handle, entry))
                    save()
                while not failed and any(entry["status"] == "running" for _, _, entry in active):
                    changed = False
                    for process, _, entry in active:
                        if entry["status"] != "running":
                            continue
                        code = process.poll()
                        if code is not None:
                            entry.update(status="complete" if code == 0 else "failed", returncode=code)
                            failed |= code != 0
                            changed = True
                    if changed:
                        save()
                    if not failed and any(entry["status"] == "running" for _, _, entry in active):
                        time.sleep(0.1)
                if failed:
                    stop_owned(active, "another command in this group failed")
            except BaseException:
                stop_owned(active, "runner interrupted or orchestration failed")
                raise
            finally:
                for _, handle, _ in active:
                    handle.close()
            if failed:
                raise PipelineFailure(f"Stage group {group.name} failed; no later group was started. Logs: {logs}")
        report["status"] = "complete"
        report.pop("active_group", None)
        save()
        return report
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True, help="JSON settings, e.g. configs/olmo3.json")
    result.add_argument("--corpus", required=True, help="Original JSONL/Parquet corpus")
    result.add_argument("--queries", required=True, help="Canonical validation query JSONL")
    result.add_argument("--output", required=True, help="Run root; individual stage outputs must be new")
    result.add_argument("--model", help="Override config model with a local snapshot or Hub ID")
    result.add_argument("--gpus", default="0", help="Distinct physical GPU indices, e.g. 0,1")
    result.add_argument("--backend", choices=["transformers", "vllm"], help="Override config backend; default vllm")
    result.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    result.add_argument("--external-labels", help="Rewards for the actual externally evaluated rollouts")
    result.add_argument("--dry-run", action="store_true", help="Only print commands; do not create outputs or start children")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = read_config(args.config)
        plan = build_plan(args, config)
        if args.dry_run:
            for group in plan:
                print(f"# {group.name}")
                for command in group.commands:
                    print(command.display())
            return 0
        validate_all_verifiers(args)
        metadata = {"config_file": str(args.config), "config": config,
                    "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                    "requested_stage": args.stage, "model_override": args.model,
                    "backend_override": args.backend, "gpus": parse_gpus(args.gpus),
                    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        report = execute_plan(plan, args.output, metadata=metadata)
        print(json.dumps({"status": report["status"], "logs": report["logs"]}, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Runner interrupted; its owned child groups were stopped. Inspect the retained status and logs.", file=sys.stderr)
        return 130
    except (OSError, ValueError, PipelineFailure) as error:
        print(f"run_method: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
