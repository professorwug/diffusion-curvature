# Zetteldev project task runner
# Run `just` to see available commands

default:
    @just --list

# === Development ===

# Run pytest
test:
    pixi run pytest

# Install ipykernel for this project
postinstall:
    pixi run python -m ipykernel install --user --name "diffusion_curvature-zetteldev"

# Verify the project module imports correctly
test-import:
    pixi run python -c "import diffusion_curvature"

# === Notebooks ===

# Launch Jupyter Lab
notebooks:
    pixi run jupyter lab --ip=0.0.0.0

# Launch Jupyter Notebook (CUDA environment)
cuda-jupyter:
    pixi run -e cuda jupyter notebook

# === Marimo ===

# Launch Marimo in edit mode
mo *args:
    pixi run marimo edit {{args}} --watch --headless --port 8642 --token-password rahlsins --host 0.0.0.0

# Launch Marimo in run mode
marimo-run:
    pixi run marimo run --headless --port 8642 --token-password themediumisthemessage --proxy athomia.moose-walleye.ts.net --base-url /marimo

# === NBdev ===

export QUARTO_PYTHON := justfile_directory() / ".pixi/envs/default/bin/python"

# Render a Jupyter notebook to PDF (strips NBDev directives first)
render-notebook input output:
    pixi run python .zetteldev/render_notebook.py {{input}} {{output}}

# Export notebooks to Python modules
nbsync:
    pixi run nbdev_export

# Clean notebook metadata
nbclean:
    pixi run nbdev_clean

nbprep:
    pixi run nbdev_prepare

# Update notebooks from Python modules
pysync:
    pixi run nbdev_update

# Generate documentation
docmaker:
    pixi run nbdev_docs

# Render notebooks to zettels (markdown)
zettelmaker:
    pixi run quarto render experiments --to gfm --no-execute --output-dir ../zettels --profile zettels

# === Experiments ===

# Run snakemake (pass args like: just snakemake all)
snakemake *args:
    pixi run snakemake {{args}}

# Create a new experiment
create-experiment:
    pixi run python .zetteldev/create_experiment.py

# Run an experiment
run-experiment:
    pixi run python .zetteldev/run_experiment.py

# === Claude Code ===

# Start Claude with experiment-specific task list (auto-detects from cwd)
# Pass any args to claude, e.g.: just claude --resume
[no-cd]
claude *args:
    #!/usr/bin/env bash
    invocation_dir="{{invocation_directory()}}"
    if [[ "$invocation_dir" =~ experiments/([^/]+) ]]; then
        task_id="diffusion-curvature-${BASH_REMATCH[1]}"
        echo "Task list: $task_id"
        CLAUDE_CODE_TASK_LIST_ID="$task_id" claude --dangerously-skip-permissions {{args}}
    else
        echo "Starting Claude without persistent task list (not in experiment dir)"
        claude --cwd "$invocation_dir" --dangerously-skip-permissions {{args}}
    fi

# === Cloud Sync ===

# Sync data to cloud storage
sync-to-cloud:
    pixi run python .zetteldev/sync_cloud_data.py sync-to-cloud

# Sync data from cloud storage
sync-from-cloud:
    pixi run python .zetteldev/sync_cloud_data.py sync-from-cloud

# === Hugging Face Data Storage ===

# Manage experiment data on Hugging Face
# Usage:
#   just hugdata status              - show sync status for all experiments
#   just hugdata push <experiment>   - push experiment to HF (versioned snapshot)
#   just hugdata pull <path>         - pull from HF (supports subdirs)
#   just hugdata pull --all          - pull all experiments
#   just hugdata init                - initialize HF repo connection
hugdata action *args:
    pixi run python .zetteldev/hf_data.py {{action}} {{args}}

# Push all local-only and ahead experiments to HF
hugdata-pushitall:
    pixi run python .zetteldev/hf_data.py pushall

# === Zetteldev Container ===

# Update zetteldev assets
zdev-update:
    bash .zetteldev/update_zetteldev_assets.sh

# Start a shell in the zdev container
zdev-shell:
    bash .zetteldev/zdev-container.sh shell

# Start Claude in the zdev container
zdev-claude:
    bash .zetteldev/zdev-container.sh claude

# Start Codex in the zdev container
zdev-codex:
    bash .zetteldev/zdev-container.sh codex

# Stop the zdev shell container
zdev-stop-shell:
    bash .zetteldev/zdev-container.sh stop shell

# Remove the zdev shell container
zdev-rm-shell:
    bash .zetteldev/zdev-container.sh rm shell

# === Della Sync ===

# Sync processed_data from Della for current experiment (auto-detected from cwd)
# Usage:
#   cd experiments/18-psrl-empowerment-sampling && just dellaloot
#   just dellaloot all                    # sync all experiments
#   just dellaloot --dry-run              # preview what would sync
#   just dellaloot --exclude="*.inprogress"  # pass extra rsync flags
dellaloot *args:
    #!/usr/bin/env bash
    set -euo pipefail
    root="{{justfile_directory()}}"
    host=$(python3 -c "import tomllib; print(tomllib.load(open('$root/.zetteldev/della.toml','rb'))['cluster']['host'])")
    remote=$(python3 -c "import tomllib; print(tomllib.load(open('$root/.zetteldev/della.toml','rb'))['cluster']['remote_path'])")
    sync_dirs=$(python3 -c "import tomllib; print(' '.join(tomllib.load(open('$root/.zetteldev/della.toml','rb'))['cluster']['sync_dirs']))")

    sync_experiment() {
        local exp="$1"; shift
        for dir in $sync_dirs; do
            local src="$host:$remote/experiments/$exp/$dir/"
            local dst="$root/experiments/$exp/$dir/"
            mkdir -p "$dst"
            echo "⟵  experiments/$exp/$dir/"
            rsync -avz --exclude="*.inprogress" --progress "$@" "$src" "$dst"
            echo ""
        done
    }

    # Check if "all" was passed as first arg
    first_arg="${1:-}"
    if [[ "$first_arg" == "all" ]]; then
        shift || true
        for exp_dir in "$root"/experiments/*/; do
            exp=$(basename "$exp_dir")
            # Only sync experiments that have a processed_data dir locally or remotely
            for dir in $sync_dirs; do
                if [[ -d "$exp_dir/$dir" ]]; then
                    sync_experiment "$exp" "$@"
                    break
                fi
            done
        done
    else
        invocation_dir="{{invocation_directory()}}"
        if [[ "$invocation_dir" =~ experiments/([^/]+) ]]; then
            sync_experiment "${BASH_REMATCH[1]}" "$@"
        else
            echo "Error: not in an experiment directory. Use 'just dellaloot all' or cd into an experiment."
            exit 1
        fi
    fi

# === Cluster Monitoring ===

# GPU dashboard for Della cluster (SSH from local machine)
jobstats *args:
    python3 .zetteldev/gpu_dashboard.py {{args}}

# === Figure Publishing ===

# Publish a figure to Cloudflare R2 (returns a stable URL for Obsidian)
# Usage:
#   just publish-figure experiments/20-foo/figures/fig_bar.png   # single figure
#   just publish-figure experiments/20-foo                       # all in experiment
#   just publish-figure <path> --obsidian                        # markdown embed
#   just publish-figure <path> --dry-run                         # preview
publish-figure *args:
    pixi run python .zetteldev/publish_figure.py {{args}}

# Sync all changed figures across all experiments to R2
publish-figures *args:
    pixi run python .zetteldev/publish_figure.py --sync {{args}}

# === Git Worktrees ===

# Create a new git worktree
pug:
    bash .zetteldev/new_git_worktree.sh

# Review a git worktree
rug:
    bash .zetteldev/review_worktree.sh

# Lazygit
lg:
    lazygit
