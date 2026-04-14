# Shared Hydra Configs

Reusable Hydra config groups for experiments that need cluster training.

## Launcher configs

SLURM launcher presets for the Della cluster, organized by partition and GPU count.
Resources scale at 6 CPUs and 64GB RAM per GPU.

Use with `hydra/launcher=<name>` (via symlink from your experiment's `conf/hydra/launcher/`).

### pli-c (H100 80GB, 5hr default)

| Config | GPUs | CPUs | RAM | Usage |
|--------|------|------|-----|-------|
| `pli_c_1gpu` | 1 | 6 | 64GB | Eval, small training |
| `pli_c_2gpu` | 2 | 12 | 128GB | Medium training |
| `pli_c_3gpu` | 3 | 18 | 192GB | Most training runs |
| `pli_c_4gpu` | 4 | 24 | 256GB | Large batch / 32-gen |

### ailab (H200 141GB, 9hr default)

| Config | GPUs | CPUs | RAM | Usage |
|--------|------|------|-----|-------|
| `ailab_1gpu` | 1 | 6 | 64GB | Eval |
| `ailab_2gpu` | 2 | 12 | 128GB | DDP training |
| `ailab_3gpu` | 3 | 18 | 192GB | Training |
| `ailab_4gpu` | 4 | 24 | 256GB | Large training |

## Environment configs

Cluster-specific environment variables. Include via `defaults: [/env: della]`.

| Config | Sets |
|--------|------|
| `della` | WANDB_MODE=offline, CUDA alloc, cuDNN SDPA disable |

## Usage in experiments

Set up the symlink (done automatically by `just create-experiment --hydra`):
```bash
cd experiments/my-experiment
ln -s ../../.zetteldev/hydra/launcher conf/hydra/launcher_shared
```

Then from the command line:
```bash
# Submit to pli-c with 3 GPUs
python train.py -m hydra/launcher=launcher_shared/pli_c_3gpu

# Submit to ailab with 4 GPUs
python train.py -m hydra/launcher=launcher_shared/ailab_4gpu

# Sweep on pli-c with 3 GPUs
python train.py -m lr=1e-5,1e-6 hydra/launcher=launcher_shared/pli_c_3gpu

# Override time limit
python train.py -m hydra/launcher=launcher_shared/pli_c_3gpu hydra.launcher.timeout_min=60
```

## Adding a new config

Create `launcher/<partition>_<N>gpu.yaml`:

```yaml
# @package hydra.launcher
_target_: hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher
partition: <partition-name>
gpus_per_node: <N>
cpus_per_task: <N * 6>
mem_gb: <N * 64>
timeout_min: 300
tasks_per_node: 1
nodes: 1
```
