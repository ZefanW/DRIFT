import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('continue_sft', Path(__file__).parents[1] / 'scripts/continue_sft.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def args(*extra):
    return module.parser().parse_args(['--model', 'local/model', '--dataset', 'subset.jsonl', '--output', 'new-run', *extra])


def test_gpu_count_preserves_global_token_budget():
    for devices, accumulation in [('0,1,2,3,4,5,6,7', '1'), ('0,1,2,3', '2'), ('0', '8')]:
        cmd, env = module.command_for(args('--gpus', devices))
        assert cmd[cmd.index('--gradient_accumulation_steps')+1] == accumulation
        assert env['NPROC_PER_NODE'] == str(len(devices.split(',')))
        assert env['CUDA_VISIBLE_DEVICES'] == devices


@pytest.mark.parametrize('extra', [('--gpus', '0,1,2'), ('--gpus', '0,0'), ('--attn-impl', 'sdpa'), ('--max-steps', '0')])
def test_invalid_recipe_fails(extra):
    with pytest.raises(ValueError):
        module.command_for(args(*extra))


def test_dry_run_is_side_effect_free(tmp_path):
    out = tmp_path / 'does-not-exist'
    assert module.main(['--model', 'does-not-exist', '--dataset', 'does-not-exist', '--output', str(out), '--dry-run']) == 0
    assert not out.exists()


def test_paper_recipe_matches_recorded_olmo3_settings():
    cmd, _ = module.command_for(args())
    expected = {'--template': 'dummy', '--template_backend': 'jinja',
                '--truncation_strategy': 'delete', '--adam_beta2': '0.95',
                '--adam_epsilon': '1e-8', '--weight_decay': '0', '--use_liger_kernel': 'true'}
    for key, value in expected.items():
        assert cmd[cmd.index(key)+1] == value
    assert '--external_plugins' not in cmd


def test_assistant_only_training_is_explicitly_separate():
    cmd, _ = module.command_for(args('--template-mode', 'assistant'))
    assert cmd[cmd.index('--template')+1] == 'drift_chat'
    assert '--external_plugins' in cmd
