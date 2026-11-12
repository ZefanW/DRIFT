#!/usr/bin/env python3
"""Run the paper's eight OLMo3 evaluations in separate, patched environments.

Use --dry-run to print commands without loading a model or creating files.
See evaluation/README.md for the required upstream commits and patches.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

EVALUATION = Path(__file__).resolve().parents[1] / 'evaluation'


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, help='Local OLMo3 checkpoint directory')
    p.add_argument('--output', required=True, help='New output directory; never overwritten')
    p.add_argument('--olmes-python', required=True)
    p.add_argument('--lighteval-python', required=True)
    p.add_argument('--olmes-source', required=True, help='Patched OLMES checkout root')
    p.add_argument('--lighteval-source', required=True, help='Patched LightEval checkout root')
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--preflight-only', action='store_true', help='Check sources/environments; no evaluation')
    return p


def commands(args):
    devices = args.gpus.split(',')
    if not devices or any(not x.isdigit() for x in devices):
        raise ValueError('--gpus must contain comma-separated physical GPU indices')
    devices = [str(int(x)) for x in devices]
    if len(devices) != len(set(devices)):
        raise ValueError('--gpus must not repeat a device')
    checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    if any(c in checkpoint for c in ',{}\n\r'):
        raise ValueError('Checkpoint path contains an evaluator argument delimiter')
    output = Path(args.output).expanduser().resolve()
    common = {'CUDA_VISIBLE_DEVICES': ','.join(devices), 'FINAL': 'true', 'PP': '1', 'DP': '1',
              'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO': '0'}
    light = (f'model_name={checkpoint},dtype=bfloat16,tensor_parallel_size={len(devices)},'
             'max_model_length=32768,gpu_memory_utilization=0.8,seed=1234,'
             'generation_parameters={max_new_tokens:4096,temperature:TEMP,top_p:0.95,seed:42},'
             'trust_remote_code=true')
    stages = []
    for name, task, temperature in [
        ('lighteval_sampled', 'olympiad_bench:OE_TO_maths_en_COMP,gpqa:diamond', '0.6'),
        ('lighteval_mmlu', 'mmlu_pro', '0.0'),
    ]:
        stages.append({'name': name, 'engine': 'lighteval', 'output': str(output / name),
                       'command': [str(Path(args.lighteval_python).expanduser().absolute()),
                                   '-m', 'lighteval', 'vllm', light.replace('TEMP', temperature),
                                   task, '--output-dir', str(output / name), '--save-details']})
    stages.append({'name': 'olmes', 'engine': 'olmes', 'output': str(output / 'olmes'),
                   'command': [str(Path(args.olmes_python).expanduser().absolute()),
                               '-m', 'oe_eval.launch', '--output-dir', str(output / 'olmes'),
                               '--model', checkpoint, '--task', 'olmo3:adapt_0424',
                               '--model-type', 'vllm', '--batch-size', '64', '--model-args',
                               json.dumps({'trust_remote_code': 'true', 'max_length': 4096}),
                               '--recompute-metrics']})
    return stages, common


def stage_environment(args, engine, common):
    source = Path(getattr(args, engine + '_source')).expanduser().resolve()
    interpreter = Path(getattr(args, engine + '_python')).expanduser().absolute()
    env = {**os.environ, **common}
    for key in ['TEMPERATURE', 'REPEATS', 'PYTHONHOME']:
        env.pop(key, None)
    env['PYTHONPATH'] = str(source / 'src' if engine == 'lighteval' else source)
    # OLMES launch.py invokes bare `python` for its worker.
    env['PATH'] = str(interpreter.parent) + os.pathsep + env.get('PATH', '')
    return env


def fingerprint(args, engine, common):
    source = Path(getattr(args, engine + '_source')).expanduser().resolve()
    contract = read_json(EVALUATION / 'sources.json')[engine]
    observed = {}
    for name, expected in contract['files_sha256'].items():
        actual = sha256(source / name)
        if actual != expected:
            raise ValueError(f'{engine} source mismatch: {name}; apply the documented patch to the pinned checkout')
        observed[name] = actual
    module = 'oe_eval' if engine == 'olmes' else 'lighteval'
    probe = """import importlib.metadata as m, importlib.util, json, platform, sys
s = importlib.util.find_spec(sys.argv[1])
print(json.dumps({'python':platform.python_version(), 'executable':sys.executable,
 'module_origin':None if s is None else s.origin,
 'packages':sorted((d.metadata.get('Name',''),d.version) for d in m.distributions())}))
"""
    result = subprocess.run([getattr(args, engine + '_python'), '-c', probe, module],
                            env=stage_environment(args, engine, common), check=True,
                            capture_output=True, text=True)
    info = json.loads(result.stdout)
    if not info['module_origin'] or not Path(info['module_origin']).resolve().is_relative_to(source):
        raise ValueError(f'{engine} import resolves outside the requested checkout')
    names = {name.lower().replace('_', '-'): version for name, version in info['packages']}
    if 'vllm' not in names:
        raise ValueError(f'{engine} environment does not contain vllm')
    if engine == 'lighteval' and names.get('lighteval') != '0.13.0':
        raise ValueError('Install the pinned LightEval 0.13.0 checkout in its environment')
    return {**info, 'files_sha256': observed, 'upstream_commit': contract['upstream_commit']}


def checked_score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'Invalid evaluation score: {value!r}')
    return value


def validate_results(stage, protocol, checkpoint, gpu_count):
    """Require all expected questions/responses and explicit metric names."""
    root = Path(stage['output'])
    scores = {}
    if stage['engine'] == 'olmes':
        source = root / 'metrics.json'
        data = read_json(source)
        config = data['model_config']
        if Path(config['model']).resolve() != checkpoint or config['max_length'] != 4096:
            raise ValueError('OLMES checkpoint/context differs from the requested run')
        tasks = data['tasks']
        aliases = [t['alias'] for t in tasks]
        if len(aliases) != len(set(aliases)):
            raise ValueError('Duplicate OLMES task aliases')
        for prefix, expected_parts in [('bbh', protocol['bbh_subtasks']), ('minerva_math', protocol['math_subtasks'])]:
            parts = [t for t in tasks if t['alias'].startswith(prefix + '_')]
            if {t['alias'].split(':')[0] for t in parts} != {prefix + '_' + n for n in expected_parts}:
                raise ValueError(f'Unexpected {prefix} subtask coverage')
        for name, expected in protocol['tasks'].items():
            if expected['engine'] != 'olmes':
                continue
            task = next(t for t in tasks if t['alias'] == expected['alias'])
            tc = task['task_config']
            gen = tc['generation_kwargs']
            if (task['num_instances'], gen.get('repeats'), gen.get('temperature')) != (
                    expected['instances'], expected['responses'], expected['temperature']):
                raise ValueError(f'Unexpected count or decoding parameters: {name}')
            if 'aggregation' in expected and tc.get('primary_metric') != expected['aggregation']:
                raise ValueError(f'Unexpected averaging rule: {name}')
            value = checked_score(task['metrics'][expected['metric']])
            if name in ['bbh', 'minerva_math']:
                parts = [t for t in tasks if t['alias'].startswith(name + '_')]
                n = sum(t['num_instances'] for t in parts)
                if n != expected['instances']:
                    raise ValueError(f'Incomplete subtask counts: {name}')
                aggregate = (sum(checked_score(t['metrics']['primary_score']) * t['num_instances'] for t in parts) / n
                             if name == 'bbh' else sum(checked_score(t['metrics']['primary_score']) for t in parts) / len(parts))
                if not math.isclose(value, aggregate, abs_tol=1e-12):
                    raise ValueError(f'Aggregate differs from subtask results: {name}')
            scores[name] = {'score': value, 'source': str(source), 'source_sha256': sha256(source)}
    else:
        import pyarrow.parquet as pq
        names = ['gpqa', 'olympiad_bench'] if stage['name'] == 'lighteval_sampled' else ['mmlu_pro']
        files = [(f, read_json(f)) for f in root.glob('results/**/results_*.json')]
        for name in names:
            expected = protocol['tasks'][name]
            key = expected['task_key']
            matches = [(f, d) for f, d in files if key in d.get('results', {})]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one results source for {name}')
            source, data = matches[0]
            value = checked_score(data['results'][key][expected['metric']])
            config = data['config_general']['model_config']
            gen = config['generation_parameters']
            if Path(config['model_name']).resolve() != checkpoint:
                raise ValueError(f'Unexpected checkpoint: {name}')
            if (gen.get('temperature'), gen.get('seed'), gen.get('max_new_tokens'), gen.get('top_p')) != (
                    expected['temperature'], 42, 4096, .95):
                raise ValueError(f'Unexpected generation settings: {name}')
            if config.get('seed') != 1234 or config.get('tensor_parallel_size') != gpu_count:
                raise ValueError(f'Unexpected model seed or parallelism: {name}')
            if data['config_tasks'][key].get('num_fewshots') != 0:
                raise ValueError(f'Unexpected few-shot count: {name}')
            details = list(root.glob('details/**/details_' + key + '_*.parquet'))
            if len(details) != 1:
                raise ValueError(f'Expected exactly one detail file for {name}')
            parquet = pq.ParquetFile(details[0])
            if parquet.metadata.num_rows != expected['instances']:
                raise ValueError(f'Unexpected question count: {name}')
            for batch in parquet.iter_batches(batch_size=128, columns=['model_response.text']):
                for row in batch.to_pylist():
                    if len(row['model_response']['text']) != expected['responses']:
                        raise ValueError(f'Unexpected response count: {name}')
            scores[name] = {'score': value, 'source': str(source), 'source_sha256': sha256(source),
                            'details': str(details[0])}
    return scores


def main(argv=None):
    args = parser().parse_args(argv)
    manifest = None
    output = Path(args.output).expanduser().resolve()
    try:
        stages, common = commands(args)
        for stage in stages:
            print(shlex.join(stage['command']), flush=True)
        if args.dry_run:
            return 0
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not (checkpoint / 'config.json').is_file():
            raise ValueError('Provide a local checkpoint containing config.json')
        if output.exists():
            raise FileExistsError('Output already exists; choose a new directory')
        # Validation must be available before spending GPU time.
        import pyarrow.parquet  # noqa: F401
        locks = {e: fingerprint(args, e, common) for e in ['olmes', 'lighteval']}
        if args.preflight_only:
            print('Source and environment checks passed; no evaluation executed.')
            return 0
        protocol = read_json(EVALUATION / 'protocol.json')
        output.mkdir(parents=True, exist_ok=False)
        manifest = {'state': 'running', 'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'checkpoint': str(checkpoint), 'checkpoint_metadata_sha256': {
                        n: sha256(checkpoint / n) for n in ['config.json', 'tokenizer_config.json',
                        'generation_config.json', 'model.safetensors.index.json'] if (checkpoint / n).is_file()},
                    'weight_identity': 'Use the accompanying training/checkpoint manifest for weight provenance.',
                    'environment_overrides': common, 'unset_environment': ['TEMPERATURE', 'REPEATS'],
                    'environments': locks, 'protocol': protocol, 'stages': stages}
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        scores = {}
        for stage in stages:
            engine = stage['engine']
            if fingerprint(args, engine, common) != locks[engine]:
                raise ValueError(f'{engine} environment changed before evaluation')
            stage['state'] = 'running'
            with (output / (stage['name'] + '.log')).open('x') as log:
                result = subprocess.run(stage['command'], env=stage_environment(args, engine, common),
                                        cwd=str(Path(getattr(args, engine + '_source')).expanduser().resolve()),
                                        stdout=log, stderr=subprocess.STDOUT)
            stage['returncode'] = result.returncode
            if result.returncode:
                raise RuntimeError(f"{stage['name']} exited {result.returncode}; inspect its log")
            if fingerprint(args, engine, common) != locks[engine]:
                raise ValueError(f'{engine} environment changed during evaluation')
            scores.update(validate_results(stage, protocol, checkpoint, len(common['CUDA_VISIBLE_DEVICES'].split(','))))
            stage['state'] = 'complete'
            (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if set(scores) != set(protocol['task_order']):
            raise ValueError('Eight-task evaluation is incomplete')
        summary = {'tasks': scores, 'all_percent': 100 * sum(v['score'] for v in scores.values()) / 8,
                   'target_percent': 100 * sum(scores[n]['score'] for n in protocol['task_order'][:4]) / 4,
                   'non_target_percent': 100 * sum(scores[n]['score'] for n in protocol['additional_tasks']) / 4}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        manifest['state'] = 'complete'
        print(f'Validated eight-task summary: {output / "summary.json"}')
        return 0
    except (OSError, ValueError, KeyError, StopIteration, ImportError, RuntimeError, subprocess.SubprocessError) as exc:
        if manifest is not None:
            manifest['state'] = 'failed'
            manifest['error'] = str(exc)
        print(f'error: {exc}', file=sys.stderr)
        return 1
    except BaseException:
        if manifest is not None:
            manifest['state'] = 'interrupted'
        raise
    finally:
        if manifest is not None:
            (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
