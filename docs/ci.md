# Optional continuous integration

The [workflow template](ci.example.yml) runs CPU tests, the synthetic demo, and a package build on Python 3.10 and 3.12. It does not download models or require GPUs. It is supplied as a template, not an active workflow.

To enable it, use GitHub credentials authorized to manage workflows, then place it at `.github/workflows/tests.yml` and commit the change:

```bash
mkdir -p .github/workflows
cp docs/ci.example.yml .github/workflows/tests.yml
```

The same checks can be run locally from the repository root:

```bash
python -m pip install -e '.[dev]'
python -m pytest
drift demo --output runs/ci-demo
python -m build
```

Use a fresh demo output directory on subsequent runs. GPU integration checks and their tested environments are recorded in [validation.md](validation.md).
