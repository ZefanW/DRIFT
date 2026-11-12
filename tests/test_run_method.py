"""Test orchestration and failure boundaries without GPU libraries or models."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_method.py"
SPEC = importlib.util.spec_from_file_location("drift_method_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def args(self, *extra):
        return runner.parser().parse_args([
            "--config", "config.json", "--corpus", "my corpus.jsonl", "--queries", "queries.jsonl",
            "--output", "runs/example", *extra,
        ])

    def test_full_plan_assigns_independent_ranks_and_matches_feature_settings(self):
        config = {"model": "default-model", "revision": "fixed-revision", "samples": 8,
                  "dimension": 16384, "projection_seed": 19, "max_length": 4096,
                  "fit_mode": "stable", "fraction": 0.1, "gradient_checkpointing": True}
        args = self.args("--gpus", "3,7", "--model", "my model", "--backend", "transformers")
        groups = runner.build_plan(args, config, python="custom-python")
        self.assertEqual([g.name for g in groups], ["prepare", "rollout", "label", "extract-corpus",
                         "merge-corpus", "extract-anchors", "merge-anchors", "score", "select"])
        rollout = groups[1].commands
        self.assertEqual([c.gpu for c in rollout], ["3", "7"])
        for rank, command in enumerate(rollout):
            self.assertEqual(command.argv[:3], ("custom-python", "-m", "drift"))
            self.assertEqual(command.argv[command.argv.index("--rank") + 1], str(rank))
            self.assertEqual(command.argv[command.argv.index("--world-size") + 1], "2")
            self.assertEqual(command.argv[command.argv.index("--model") + 1], "my model")
            self.assertIn("transformers", command.argv)
            self.assertIn("fixed-revision", command.argv)
        for group in [groups[3], groups[5]]:
            self.assertEqual(len(group.commands), 2)
            for command in group.commands:
                self.assertEqual(command.argv[command.argv.index("--projection-seed") + 1], "19")
                self.assertIn("--gradient-checkpointing", command.argv)
                self.assertEqual(command.argv[command.argv.index("--device") + 1], "cuda:0")
        for group in [groups[4], groups[6]]:
            self.assertIsNone(group.commands[0].gpu)
            self.assertEqual(group.commands[0].argv.count("--shards"), 1)
        self.assertIn("my corpus.jsonl", groups[0].commands[0].argv)

    def test_stage_selection_does_not_rerun_prior_stages(self):
        args = self.args("--stage", "extract", "--gpus", "0,1,2")
        groups = runner.build_plan(args, {"model": "m"})
        self.assertEqual([g.name for g in groups], ["extract-corpus", "merge-corpus", "extract-anchors", "merge-anchors"])
        self.assertEqual(len(groups[0].commands), 3)
        label = runner.build_plan(self.args("--stage", "label", "--external-labels", "labels.jsonl"), {"model": "m"})
        self.assertEqual(len(label), 1)
        self.assertIn("labels.jsonl", label[0].commands[0].argv)

    def test_public_configs_generate_commands_accepted_by_the_actual_cli(self):
        from drift.cli import parser as cli_parser

        for name in ("olmo3.json",):
            with self.subTest(config=name):
                config = runner.read_config(SCRIPT.parents[1] / "configs" / name)
                groups = runner.build_plan(self.args("--gpus", "0,1"), config)
                for group in groups:
                    for command in group.commands:
                        # Parsing only: no input reads, outputs, or model imports.
                        parsed = cli_parser().parse_args(command.argv[3:])
                        self.assertEqual(parsed.output, str(command.output))

    def test_dry_run_creates_nothing_and_starts_no_children(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({"model": "model"}))
            output = root / "not-created"
            with mock.patch.object(runner.subprocess, "Popen", side_effect=AssertionError("Unexpected process")), redirect_stdout(io.StringIO()) as stream:
                result = runner.main(["--config", str(config), "--corpus", "not-read.jsonl", "--queries", "not-read.jsonl",
                                      "--output", str(output), "--gpus", "0,2", "--dry-run"])
            self.assertEqual(result, 0)
            self.assertFalse(output.exists())
            self.assertIn("CUDA_VISIBLE_DEVICES=2", stream.getvalue())
            self.assertIn("-m drift merge", stream.getvalue())

    def test_failed_rank_aborts_owned_sibling_but_not_an_unrelated_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "merge-must-not-run"
            commands = (
                runner.Command("rank-0", (sys.executable, "-c", "print('rank failed'); raise SystemExit(7)"), root / "rank-0", "0"),
                runner.Command("rank-1", (sys.executable, "-c", "import time; time.sleep(60)"), root / "rank-1", "1"),
            )
            merge = runner.Command("merge", (sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).touch()", str(marker)), root / "merged")
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
            started = time.monotonic()
            try:
                with self.assertRaises(runner.PipelineFailure):
                    runner.execute_plan([runner.CommandGroup("extract", commands), runner.CommandGroup("merge", (merge,))], root)
                self.assertIsNone(unrelated.poll())
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=5)
            self.assertLess(time.monotonic() - started, 10)
            self.assertFalse(marker.exists())
            status_path, = (root / "runner_logs").glob("*/status.json")
            status = json.loads(status_path.read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["results"][0]["returncode"], 7)
            self.assertEqual(status["results"][1]["status"], "aborted")
            self.assertLess(status["results"][1]["returncode"], 0)
            self.assertEqual(len(status["results"]), 2)
            self.assertIn("rank failed", Path(status["results"][0]["log"]).read_text())
            self.assertTrue(Path(status["results"][1]["log"]).is_file())

    def test_existing_output_blocks_entire_plan_before_any_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing"
            existing.mkdir()
            command = runner.Command("existing", ("must-not-run",), existing)
            with mock.patch.object(runner.subprocess, "Popen", side_effect=AssertionError("Unexpected process")), self.assertRaises(FileExistsError):
                runner.execute_plan([runner.CommandGroup("stage", (command,))], root)
            self.assertFalse((root / "runner_logs").exists())

    def test_all_requires_external_labels_before_launching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queries = root / "queries.jsonl"
            queries.write_text(json.dumps({"id": "q", "verifier": {"type": "external"}}) + "\n")
            args = self.args("--queries", str(queries))
            with self.assertRaisesRegex(ValueError, "external verifiers"):
                runner.validate_all_verifiers(args)
            labels = root / "labels.jsonl"
            labels.write_text('{"id":"q:0","reward":1}\n')
            args.external_labels = str(labels)
            runner.validate_all_verifiers(args)

    def test_invalid_gpu_list_and_unknown_config_are_rejected(self):
        for value in ["", "0,", "0,0", "01,1", "-1", "0,a"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.parse_gpus(value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"model":"m","silently_ignored":true}')
            with self.assertRaisesRegex(ValueError, "Unsupported configuration"):
                runner.read_config(path)


if __name__ == "__main__":
    unittest.main()
