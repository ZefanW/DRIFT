#!/usr/bin/env python3
"""Continue full SFT using the public ms-swift trainer; never overwrite a run.

Install ms-swift in this Python environment before using this entry point.
The default recipe packs eight 32,768-token sequences per optimizer update.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--revision')
    p.add_argument('--dataset', required=True, help='Exported subset.jsonl')
    p.add_argument('--output', required=True, help='New run directory')
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--template-mode', choices=['paper', 'assistant'], default='paper',
                   help='paper preserves the original OLMo3 Jinja training objective')
    p.add_argument('--max-steps', type=int, default=500)
    p.add_argument('--max-length', type=int, default=32768)
    p.add_argument('--global-sequences', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--min-learning-rate', type=float, default=1e-6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-seed', type=int, default=42)
    p.add_argument('--attn-impl', choices=['flash_attn', 'sdpa', 'eager'], default='flash_attn')
    p.add_argument('--no-packing', action='store_true', help='Changes the token budget; intended for small checks')
    p.add_argument('--dataset-num-proc', type=int, default=32)
    p.add_argument('--dataloader-num-workers', type=int, default=32)
    p.add_argument('--master-port', type=int, default=29501)
    p.add_argument('--dry-run', action='store_true')
    return p


def command_for(args):
    devices = args.gpus.split(',')
    if not devices or any(not value.isdigit() for value in devices) or len(devices) != len(set(devices)):
        raise ValueError('--gpus must list unique, nonnegative physical GPU indices')
    devices = [str(int(value)) for value in devices]
    if len(devices) != len(set(devices)):
        raise ValueError('--gpus contains duplicate physical GPU indices')
    if args.global_sequences < 1 or args.global_sequences % len(devices):
        raise ValueError('--global-sequences must be a positive multiple of the GPU count')
    if min(args.max_steps, args.max_length, args.dataset_num_proc) < 1:
        raise ValueError('Step, context and worker counts must be positive')
    if not 0 < args.min_learning_rate <= args.learning_rate:
        raise ValueError('Require 0 < minimum learning rate <= learning rate')
    if not args.no_packing and args.attn_impl != 'flash_attn':
        raise ValueError('ms-swift packing requires FlashAttention')
    accumulation = args.global_sequences // len(devices)
    cmd = [sys.executable, '-m', 'swift.cli.main', 'sft',
           '--model', args.model, '--dataset', args.dataset,
           '--use_chat_template', 'true',
           '--output_dir', str(Path(args.output) / 'checkpoints'),
           '--train_type', 'full', '--torch_dtype', 'bfloat16',
           '--attn_impl', args.attn_impl, '--max_length', str(args.max_length),
           '--packing', str(not args.no_packing).lower(), '--loss_scale', 'default',
           '--max_steps', str(args.max_steps), '--per_device_train_batch_size', '1',
           '--gradient_accumulation_steps', str(accumulation),
           '--learning_rate', str(args.learning_rate), '--warmup_ratio', '0.1',
           '--lr_scheduler_type', 'cosine_warmup_with_min_lr',
           '--lr_scheduler_kwargs', json.dumps({'min_lr': args.min_learning_rate}),
           '--optim', 'adamw_torch_fused', '--optim_args', 'eps=1e-6',
           '--adam_beta1', '0.9', '--adam_beta2', '0.95', '--weight_decay', '0',
           '--adam_epsilon', '1e-8', '--max_grad_norm', '1', '--use_liger_kernel', 'true',
           '--gradient_checkpointing', 'true', '--seed', str(args.seed),
           '--data_seed', str(args.data_seed), '--split_dataset_ratio', '0',
           '--dataset_shuffle', 'true', '--train_dataloader_shuffle', 'true',
           '--dataset_num_proc', str(args.dataset_num_proc),
           '--dataloader_num_workers', str(args.dataloader_num_workers),
           '--save_steps', str(min(100, args.max_steps)), '--save_only_model', 'true',
           '--save_total_limit', '1', '--logging_steps', '1', '--report_to', 'none',
           '--add_version', 'false']
    if args.template_mode == 'paper':
        cmd += ['--template', 'dummy', '--template_backend', 'jinja', '--truncation_strategy', 'delete']
    else:
        cmd += ['--external_plugins', str(Path(__file__).resolve().with_name('swift_template.py')),
                '--template', 'drift_chat', '--template_backend', 'swift', '--truncation_strategy', 'right']
    if args.revision:
        cmd += ['--model_revision', args.revision]
    if len(devices) > 1:
        cmd += ['--deepspeed', 'zero2']
    env = {'CUDA_VISIBLE_DEVICES': ','.join(devices), 'NPROC_PER_NODE': str(len(devices)),
           'NNODES': '1', 'NODE_RANK': '0', 'MASTER_ADDR': '127.0.0.1',
           'MASTER_PORT': str(args.master_port), 'TOKENIZERS_PARALLELISM': 'false'}
    return cmd, env


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        cmd, env = command_for(args)
        print(shlex.join([f'{k}={v}' for k, v in env.items()]) + ' ' + shlex.join(cmd), flush=True)
        if args.dry_run:
            return 0
        from drift.io import read_jsonl, sha256_file, validate_messages
        count = 0
        for row in read_jsonl(args.dataset):
            validate_messages(row['messages'])
            if not any(message['role'] == 'assistant' for message in row['messages']):
                raise ValueError('Training conversations must contain assistant targets')
            count += 1
        if not count:
            raise ValueError('Training subset is empty')
        version = importlib.metadata.version('ms-swift')
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=False)
        manifest = {'state': 'running', 'command': cmd, 'environment_overrides': env,
                    'ms_swift_version': version, 'dataset_sha256': sha256_file(args.dataset),
                    'dataset_rows': count, 'configuration': vars(args),
                    'nominal_tokens_per_update': None if args.no_packing else args.global_sequences * args.max_length}
        path = out / 'training_manifest.json'
        path.write_text(json.dumps(manifest, indent=2) + '\n')
        try:
            with (out / 'train.log').open('x') as log:
                result = subprocess.run(cmd, env={**os.environ, **env}, stdout=log, stderr=subprocess.STDOUT)
            manifest['returncode'] = result.returncode
            manifest['state'] = 'complete' if result.returncode == 0 else 'failed'
        except BaseException:
            manifest['state'] = 'interrupted'
            raise
        finally:
            path.write_text(json.dumps(manifest, indent=2) + '\n')
        return result.returncode
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
