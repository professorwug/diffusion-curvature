# Zetteldev

This is a 'Zetteldev' repo for conducting academic research through literate programming. All experimentation and coding must be grounded by markdown design documents supplied by the user. You, programming genie as you are, have the job of being an exceptionally capable (and rather witty) *interpreter* of the user's literate programs, translating the (ideally) highly efficient and expressive combination of English, mathematical notation and pseudocode into robust and interpretable python. 

## Shorthand Commands

- **`rar`** (Reread and Run): Re-read the last active zettel (daily note or experiment zettel), find any new placeholder figures/tables/requests, create scripts to produce them (saved to well-organized python files in appropriate subdirectories), and run them. Always save code to files — never run long computations inline. When a computation is expected to take a while, save useful intermediate results (dataframes, arrays, model outputs) to `processed_data/` so they can be reused by downstream figure scripts without recomputation.

# Procedure: Phases of an Experiment

**The markdown files which guide and document each experiment are found within the user's Obsidian vault, in ~/Pumberton.** These are standard Obsidian markdown: they contain [[wikilinks]] and YAML frontmatter.

**Most work happens within `experiments/` subfolders.** You will either be asked to work within an existing experiment or create a new one.
   - If in an existing experiment, familiarize yourself with the existing code. Crucially, also ensure you have located and read the 'zettel' in Obsidian. If the user doesn't provide it, consult `design.md` for the path. If the user provides a filename, look in `~/Pumberton/Stream` - this is the default location. 
   - If creating an experiment, run `just create-experiment experiment-name`. This will scaffold a main.py, a report.qmd, and a Snakefile. For experiments that need cluster training with hyperparameter sweeps, use `just create-experiment experiment-name --hydra` to also scaffold a Hydra config directory and train.py entry point.

The user should tell you which phase you are operating within; if not it should be inferred from the context. If given a PDF, you are in Phase 1. If asked to modify some aspect of an existing experiment or to launch evaluation jobs, you are in Phase 3.

## Phase 1: Ideation


1. **You will be given the name of a zettel with the seed for an experiment.** Find this file in `~/Pumberton/Stream` or elsewhere in `~/Pumberton`. Record the location in design.md for future reference. 
   - Read the note thoroughly and also read any linked context. Each experiment begins with frontmatter that describes it as a `type: search` with `expedition: [[Parent Note Name]]`. The user will also use wikilinks in the main zettel to direct you to other relevant notes, e.g. past experiment results or a draft write-up of the project.
   - This may be a note which already has a lot of context and results. In this case, the section to be ideated is at the bottom of the zettel. Focus on this.

2. After you have thoroughly understood the design specification, familiarize yourself with the context of the current directory. What scaffolding is in place? The zettel should describe the operational mechanics of the existing code in broad strokes, but you will need to fill in the pieces.

3. Based on your understanding, have a dialogue with the user. Walk them through the areas of underspecification in the design. Air any doubts you have, or possible confusions based on the language. The design zettel should be a faithful enough description of the experiment that with it alone the experiment can be reproduced! Based on the dialogue, the user may update the design. He'll endeavor to understand the implementation in depth and describe it within an `## Implementation` subsection in the zettel. Continue this step until the updates have ceased.

4. Finally, write up your implementation plan in the `## Implementation` section. Describe the files that will be modified and created. Be terse and efficient in this description, reusing the terminology from the rest of the zettel. Describe the classes and functions to be created and use mermaid diagrams to show the dataflow. **Prefer writing plans in Obsidian to using your built-in planning tool** - Obsidian plans are easier to review, more expressive, and integrated with the context of the experiment. Put your plan in a collapsable Obsidian style callout like `> [!info]-`. 

5. Based on the plan, the user might either continue the ideation, or direct you to proceed to Phase 2.

Do not proceed to the next phase until authorized by the user.

## Phase 2: Iteration & Implementation
*In which the design is refined through a delicate feedback loop with reality.*


1. **Fully implement the design, scaling from literate program to production-ready scaffolding.** Spend some time on this part! Follow the conventions below, putting modular bits of code in python files and tying them together with the Snakefile. Put the most important code in the experiment into the main directory, in main.py - the pieces reused by other sub-experiments. Put those sub-experiments into python files in the scripts folder. Keep each self-contained and modular. 
   - Then —always— run the experiment and collect results. These may be initial 'test' results to sanity check the method, or a full computation.
   - **Compute resources:** You are working on a machine with 2x 4090 GPUs and 30 cores; you can use these freely. You can also submit up to 4 jobs of <1 hr runtime to the Della cluster to access larger GPUs or perform inference on vLLM models like Llama.

2. **Track Uncertainties and Progress** As you work, add blockers, uncertainties, and areas of underspecification to a new `## Uncertainties` subheading in the relevant section of the design zettel.

3. **Report your findings.** After results come in, update the relevant section of the design zettel with a `## Results` subheading documenting the findings.
   - Put all results in obsidian callouts.
   - Please refrain from adding analysis of the results beyond noting the most obvious takeaways (e.g. "Method A outperforms Method B on Dataset D by 25 pts."). The user will be sharing the zettel with coworkers, and it's important the main text be written entirely by himself.
   - When adding figures to the Obsidian note, publish the figure to R2 first using `just publish-figure <path>`. This uploads to Cloudflare R2 and returns a stable public URL. Use the `--obsidian` flag to get ready-to-paste markdown. Add the image to the zettel with: `![Alt text describing the provenance, naming the originating python file](https://blots.kincaid.ink/reason_reckon/{experiment}/{filename})`. Do NOT copy figures into the Obsidian vault — use the hosted URL instead. Figures at the same URL are overwritten on re-upload, so Obsidian always shows the latest version.
   - Important: if your edits to the Obsidian note are repeatedly rejected due to changing content, the user is likely editing it. Stop and wait for them to finish.
   - **Always re-read the end of the file before adding figures or results.** The user may have added content since your last read; inserting at a stale position will overwrite their work.
   - **Add figures and tables inline**, but always put your commentary on results in quote blocks (`>`), so the provenance of writing is clear. The user's prose goes in the main text; your observations go in `>` blocks.

## Phase 3: Inundation & Optimization

Many experiments require extensive compute, carried out on Princeton's clusters (described below), together with many little tweaks to the algorithm and optimizations to the compute usage.

If working in this phase, the user will give you direct instructions.

1. Begin by reading the zettel for this experiment. It has the core context: motivation, algorithm, and description of how the rest of the experiment is structured, including how the data is saved and structured.

2. **Track work with the built-in Task tools.** At session start, run `TaskList` to see current state.
   - For large evaluations, create a parent task (e.g., "Evaluate Qwen3-80B on Paprika, 150 games") with child tasks for each SLURM job
   - Job task descriptions should include: SLURM job ID, script path, partition, output location
   - Mark `in_progress` when submitted, `completed` when results are verified
   - Use `addBlockedBy` to link job tasks to their parent evaluation


---

# Experiment Structure

The repo is structured like this:

```
.
├── experiments
│   ├── 1-example-experiment
│   │   ├── report.qmd
│   │   ├── main.py
│   │   ├── Snakefile
│   │   ├── processed_data/
│   │   ├── figures/
│   │   └── tests/
│   ├── 2-concise-description
│   │   └── ...
├── {library_name}
│   ├── utils.py
│   ├── visualization.py
│   └── ...

├── pyproject.toml
├── justfile
├── .gitattributes
└── ...
```

You'll notice that each experiment folder is prepopulated with this template:
- `Snakefile`: Defines at least two rules—(1) run main.py, (2) render report.qmd. These will be explained below!
- `main.py`: Main Python script controlling the experiment's logic.
- `report.qmd`: Quarto file for generating the final PDF and rendering figures. This is only to be edited by the user.
- `processed_data/`: Holds data outputs. Synced via Git LFS.
- `figures/`: Store images, plots, or other visuals for your report
- `tests/`: Contains experiment-specific Pytest files for unit or integration testing
- `design.md`: Stub file containing the path to the Zettel for the experiment. Create as needed.

With `--hydra`, these are also created:
- `conf/config.yaml`: Base Hydra config with experiment parameters.
- `conf/run/`: Directory for run preset YAML files.
- `conf/hydra/launcher/`: Symlink to shared SLURM launcher configs.
- `train.py`: `@hydra.main` entry point for training.

Here's how each of these works.

## Snakemake & Snakefiles
The snakefile specifies a DAG of computations associated to each experiment.
- Every experiment script must have corresponding Snakefile rule.
- Rules must specify inputs, outputs, and shell/run commands. Prefer Snakemake's 'script' option, so it automatically tracks changes to the files and makes snakemake metadata accessible to the files.
- Example rule structure:
```
rule run_main:
    input:
        data = "../../data/foo.arrow" # placeholder
    output:
        # main.py outputs
    params:
        # miscellaneous arguments to script - hardcoded CONSTANTS, and other parameters set by researchers
    script:
        "main.py"
```
- Update Snakefile when adding/modifying experiment scripts.
- Always define an `all` rule so the entire experiment can be run with `uv run snakemake`.
- The exception: long-running compute jobs like vLLM inference or RL training. For these, use Hydra+submitit (see below) or sbatch scripts. Add snakemake rules as detached nodes in the DAG for bookkeeping.

## Hydra + submitit (cluster training experiments)

For experiments that train models on the Della cluster, use **Hydra** for config management and **submitit** for SLURM job submission. This replaces manual sbatch scripts with a single `python train.py` command.

### When to use Hydra vs Snakemake
- **Snakemake**: Multi-stage pipelines (data prep → inference → analysis → figures). DAG orchestration with file-based caching.
- **Hydra**: Single entry point with many config variants. Hyperparameter sweeps. SLURM job submission via submitit.
- **Both**: Snakemake orchestrates the pipeline, Hydra handles the training step within it.

### Setup
Scaffold with `--hydra` flag:
```bash
just create-experiment my-training-experiment --hydra
```
This creates:
- `conf/config.yaml` — base Hydra config
- `conf/run/` — directory for run presets
- `conf/hydra/launcher/` — symlink to shared SLURM launcher configs in `.zetteldev/hydra/`
- `train.py` — `@hydra.main` entry point stub

### Usage
```bash
# Local run
python train.py

# With a run preset
python train.py +run=my_preset

# Submit to SLURM (pli-c partition)
python train.py -m +run=my_preset hydra/launcher=pli_c

# Hyperparameter sweep across SLURM jobs
python train.py -m lr=1e-5,1e-6 batch_size=4,8 hydra/launcher=pli_c

# Print resolved config without running
python train.py +run=my_preset --cfg job
```

### Shared launcher configs
SLURM partition presets live in `.zetteldev/hydra/launcher/` and are symlinked into each experiment:
- `pli_c` — H100 GPUs, 5hr default
- `ailab` — H200 GPUs, 9hr default

Override per-job: `hydra.launcher.timeout_min=60 hydra.launcher.gpus_per_node=4`

### Run presets
Each experiment config variant is a YAML file in `conf/run/`:
```yaml
# conf/run/v6_ee.yaml
# @package _global_
name: v6-ee
lr: 5.0e-7
ee_bonus: 0.3
num_gpus: 3
```
Applied with `+run=v6_ee`. Values override the base `conf/config.yaml`.

### wandb integration
Training runs use `WANDB_MODE=offline` on airgapped compute nodes. The `wandb-osh` daemon on the login node auto-syncs runs to wandb cloud. Start it once per session:
```bash
tmux new -s wandb-osh
cd ~/src/reason_reckon/experiments/<your-experiment>
.venv/bin/wandb-osh
```

## Python Scripts
- Anything that requires substantial computation should get its own python script. The results of computations should always be saved in the `./processed_data` folder in the experiment directory.
- If an experiment's design spec calls for multiple stages of computation, put these in separate python files.
- Scripts will be called from snakemake. Define and reuse snakemake variables instead of defining cumbersome argparsing or hard-coding anything in the script.
```python
from snakemake.script import snakemake # direct access to Snakefile variables
data_input = snakemake.input.data
```
- Secrets will be loaded from a .env file. Use python's `dotenv` to make these available.

## Figures
The primary medium for communicating results is the figure. The zettel should specify figures, and may include hand-drawn sketches for reference. Iterate on these with the user, and suggest more elegant ways to display the data if any occur to you.
- Render figures as svg files and png files.
- Figures should always be created by separate python scripts than those performing compute-intensive work. This allows faster iteration on the figure design. Wire this logic into the Snakefile.
- Make all figures beautiful. They should be publication quality, with descriptive (but not jargon-laden) axis labels and tasteful color schemes (seaborn defaults are good).
- Make figure labels descriptive enough that they can be interpreted outside of the experiment.
- Prefer `great_tables` for table rendering.

### Figure Publishing (Cloudflare R2)
Figures are hosted on Cloudflare R2 for stable, cross-device access in Obsidian notes. The public URL pattern is `https://blots.kincaid.ink/reason_reckon/{experiment}/{filename}`.

**Commands:**
- `just publish-figure <path>` — upload a single figure
- `just publish-figure <path> --obsidian` — upload and print Obsidian-ready markdown
- `just publish-figures` — sync all changed figures across all experiments
- `just publish-figures --dry-run` — preview what would be uploaded

**How it works:**
- A hash-based registry (`.zetteldev/figure_registry.json`) tracks what's been published. Only changed files are re-uploaded.
- Re-uploading to the same path overwrites the object — the URL stays stable, Obsidian always shows the latest version.
- Provenance metadata (git commit, source script, experiment) is stored on each R2 object.
- New experiments include an `onsuccess` hook in their Snakefile that auto-publishes changed figures after every successful Snakemake run.
- R2 credentials are in `.env` (`CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_R2_ACCESS_KEY_ID`, `CLOUDFLARE_R2_SECRET_ACCESS_KEY`).

## Testing
Always write and perform two types of tests.

1. *Is it doing what the user's design spec wants?* While writing scripts, write corresponding tests in the `./tests` folder, using Pytest. Execute these with `just test` (runs all tests in the project) or `uv run pytest path/to/test_file.py` for a specific file. Because there are multiple experiment directories, include the current experiment name in any tests you write to prevent collisions.
2. *Does it run without errors?* In all scripts, respect a `test_run` parameter (set globally in the Snakefile) which performs only the bare minimum computation to use all bits of the code. For example, process a tiny subset of the input data; do only 2 epochs of training; use only 2 rounds of monte-carlo sampling. After implementing an experiment, set the `test_run` to true and run snakemake. Ensure the full pipeline works.

---

# Conventions

## Environment & Package Management
This project uses **uv** for Python package management and **just** as a task runner.

### uv basics
- `uv sync` - Install all dependencies from pyproject.toml
- `uv sync --group dev` - Also install dev dependencies
- `uv run python script.py` - Run a script in the virtual environment
- `uv add package` - Add a new dependency
- `uv add --dev package` - Add a dev dependency
- `uv pip install package` - pip-compatible interface for one-off installs
- `uv pip compile` / `uv pip sync` - pip-tools compatible workflow

Dependencies are defined in `pyproject.toml`. If you need a package not listed, please ask.

### just task runner
Common tasks are defined in `justfile`. Run `just` to see all available commands:
- `just test` - Run pytest
- `just notebooks` - Launch Jupyter Lab
- `just nbsync` - Export notebooks to Python modules
- `just create-experiment` - Create a new experiment
- `just render-notebook input.ipynb output.pdf` - Render notebook to PDF (strips NBDev directives)

See the full list with `just --list`.

## Code Style Guidelines

- **Python Version**: 3.11+
- **Imports**: Standard library first, third-party packages, local modules
- **Type Annotations**: Use typing for function parameters and returns
- **Documentation**: Google-style docstrings with params and returns
- **Error Handling**: Specific exceptions with proper logging
  - Don't allow things to silently fail. If something critical to the experiment (i.e. a specified model, a necessary API key) is missing, the researcher needs to be informed immediately.

# Tips 

## Della Cluster

- You have ssh access to della for submitting scripts and retrieving results. `ssh della`. 
- Project repos are under `~/src/`, e.g. `~/src/reason_reckon`.
- When syncing code changes from local to della, use git. You can safely stash changes accrued on della.
- When syncing results from della to local, use rsync. The `processed_data` folder we save results in isn't tracked by git.
- My username on della is km5839. 

Partitions for Production and test:
- I have access to 8 H100 GPUs on the pli-c partition, and 16 faster H200 GPUs on the ailab partition. Design production scripts to use the ailab partition and target H200s with 141GB of VRAM. 
- Before submitting production scripts, always submit a test script to the pli-c partition. I have higher priority here - it will run faster. Use the test job as a feedback loop.: 
- **Testing Workflow**:
  1. Create quick test script (30min, pli-c partition)
  2. Iterate until successful trajectory saved
  3. Verify trajectory coherence (check for expected resampling, curvature scores)
  4. Launch full production jobs (3h, ailab partition)

vLLM:
- When using vLLM on Della: `../../.zetteldev/della` contains a set of bash scripts that can you be used to set up vLLM inference and embedding servers with minimal boilerplate. To be used in sbatch scripts on the Della cluster, e.g.

```bash
source "$REPO_ROOT/.zetteldev/della/lib/della.sh"

# Launch main model server (uses default served_name="model")
launch_vllm_server "$MODEL_PATH" "0" 1 "$MAIN_PORT" \
    --max-model-len 16384 \
    --tool-call-parser hermes
MAIN_PID=$VLLM_PID

# Launch embedding server (name must match config's embedding_model field!)
launch_embedding_server "$EMBED_MODEL_PATH" "1" "$EMBED_PORT" "embedding"
```

see ../../.zetteldev/della/README.md and the examples/ folder.

Tool call parsers by model family:
- Qwen models: `--tool-call-parser hermes`
- Llama models: `--tool-call-parser llama3_json`
  - **Known issue**: Llama 3.3's chat template enforces single tool call per message (`message.tool_calls|length == 1`). This can cause failures even when providing one tool at a time if history accumulates multiple tool-call messages.

Model weights location:
- Cached models live at `/scratch/gpfs/HENDERSON/transformer_cache/`
- Verify exact directory names before use (e.g., `Llama-3.1-8B-Instruct` not `Meta-Llama-3.1-8B-Instruct`)
- Common models available:
  - `Llama-3.3-70B-Instruct` (TP=2 recommended)
  - `Llama-3.1-8B-Instruct` (also used for embeddings)
  - `Qwen2.5-3B-Instruct`, `Qwen2.5-7B-Instruct`, `Qwen2.5-32B-Instruct`
  - `Qwen3-Next-80B-A3B-Instruct` (TP=2 recommended)

Della library functions:
- `launch_vllm_server MODEL_PATH GPUS TP_SIZE PORT [SERVED_NAME] [EXTRA_ARGS...]` — sets `$VLLM_PID`
  - Default `SERVED_NAME="model"` (matches with config using `openai/model`)
- `launch_embedding_server MODEL_PATH GPU PORT [SERVED_NAME] [EXTRA_ARGS...]` — sets `$EMBED_PID`
  - Default `SERVED_NAME=$(basename MODEL_PATH)`
  - **Important**: SERVED_NAME must match config's `embedding_model` field exactly
  - If config says `embedding_model: embedding`, use: `launch_embedding_server ... "embedding"`
  - Mismatch causes: `openai.NotFoundError: The model 'embedding' does not exist`
- `wait_for_server URL NAME TIMEOUT_SECONDS` — polls health endpoint

Model name resolution:
- **Working pattern** (from experiment 9):
  - vLLM server: registers as `"model"` (no prefix) — this is the default in della library
  - Config file: requests `"openai/model"` (WITH prefix)
  - liteLLM: sees "openai/" → recognizes provider → strips prefix → requests "model" → MATCH!
- This leverages liteLLM's prefix-stripping behavior as a feature
- Always explicitly set embedding server names: `launch_embedding_server ... "Llama-3.1-8B-Instruct"`

Common errors:
- `HFValidationError: Repo id must be in the form...` — model path doesn't exist, check transformer_cache
- `litellm.NotFoundError: The model 'X' does not exist` — missing `--served-model-name`, use della library functions
- `litellm.UnsupportedParamsError` — passing OpenAI-specific params (e.g., `reasoning_effort`) to local models; make these conditional

Multi-model experiments:
- All vLLM servers register as `"model"` (default)
- All configs use `player_model: openai/model` (same across all configs)
- Differentiate by output directory in config files (e.g., `processed_data/paprika-rar-qwen3b/`)
- Use bash process parallelism (`&`) not Python multiprocessing when experiments spawn their own workers

SLURM tips:
- Request conservative time limits (3h vs 6-8h) for better queue priority. Favor many short jobs over one long job.
- Test on pli-c partition first (30min, 1 game), then production on ailab (3h, 150 games)

Parallel worker pattern for long-running jobs:
- Use NUM_WORKERS=7 bash workers (not Python multiprocessing)
- Stagger start: worker 0 immediately, workers 1-6 after 600s delay
- Stagger allows JAX/GPU initialization before resource competition
- Each worker gets subset of game IDs via round-robin assignment

### Job Monitoring with /loop

After submitting SLURM jobs, **always** set up a `/loop` to monitor progress automatically. This replaces manual polling and ensures you catch failures early.

**Standard monitoring pattern:**
```
/loop 15m ssh della 'squeue -u km5839 --format="%i %j %T %M %l" 2>/dev/null; cd /scratch/gpfs/HENDERSON/km5839/reason_reckon/experiments/<EXP_DIR> && <count completed outputs>'
```

The monitoring command should:
1. Show job queue status (`squeue`)
2. Count completed outputs per condition/rollout (e.g. `find ... -name "*.json" | wc -l`)

**When to set up monitoring:**
- Immediately after `sbatch` submission
- After resubmitting mop-up jobs

**What to do at each check:**
- Report progress to the user concisely (counts + rate estimate)
- If jobs have exited the queue, check logs for errors
- If jobs completed, clean stale `.inprogress` locks and resubmit if incomplete
- Cancel jobs that have finished their work early (e.g. CS finishing before MM in a shared allocation)

**Auto-resubmit pattern** for jobs that won't finish in one allocation:
```bash
# Run in background: wait for job to end, clean locks, resubmit
ssh della 'while squeue -u km5839 --noheader --format="%i" | grep -q <JOBID>; do sleep 30; done && \
  find <output_dir> -name "*.inprogress" -delete && \
  ENV_NAME=<env> sbatch <sbatch_script>'
```

Mop-up incomplete jobs:
- Jobs often timeout before completing all games
- Use `--resume` flag in run scripts to continue from saved results
- Pattern: check game counts → resubmit if < expected → repeat
- Use `/loop` monitoring + background auto-resubmit for hands-off operation

### Shared Embedding Server

A long-running embedding server can be shared across multiple jobs via SSH tunneling, eliminating 2-3 min startup per job and saving 1 GPU per job:

- **Start once per session**: `sbatch .zetteldev/della/scripts/embed_server.sbatch`
- **Jobs connect via**: `connect_shared_embedding` (from della library)
- **Fixed port**: 12851, job name: `embed-server-$USER`
- **Check status**: `squeue -u $USER --name=embed-server-$USER`

**CRITICAL: Use --preserve-shared flag**

When using shared embedding server, ALWAYS use the `--preserve-shared` flag:
```bash
della_vllm_setup --preserve-shared
```

Without this flag, `kill_orphaned_vllm` will kill the shared embedding server's vLLM process if your job lands on the same node. This causes silent failures where the embed server bash script appears running but the vLLM process is dead.

**SSH Tunneling**

SSH tunneling works between compute nodes across partitions:
- pli-c nodes can tunnel to other pli-c nodes
- ailab nodes can tunnel to pli-c nodes (cross-partition works)
- The della library handles tunnel setup automatically via `connect_shared_embedding`

### Model Partition Requirements

- **gpt-oss-120b (TP=2)**: Prefer pli-c for faster iteration and shorter queue times. Only needs 2 H100s.
- **Qwen3-80B models** (Instruct, Thinking): Require ailab partition (H200 GPUs with 141GB VRAM)
- **Qwen3-30B, Qwen2.5-7B**: Can run on pli-c
- **GLM-4.7-FP8 (TP=4)**: Can run on pli-c with 4 GPUs
- **Kimi-K2 (TP=8)**: Requires full ailab node (8 GPUs)

### Health Checks for Long-Running Servers

vLLM servers can crash while parent bash scripts keep running. The embed_server.sbatch includes health checks that log warnings but don't auto-restart. Monitor with:
```bash
tail -f logs/embed_server_*.out | grep -i warning
```

### Experiment 15: Curvature Compass Best Practices

This experiment runs iterative 20 Questions games with ORC curvature feedback. It uses a shepherd script for multi-round automation and requires careful management of the embedding server.

#### Directory Structure
```
experiments/15-curvature-as-compass/
├── config/compass-gpt-oss-120b-{v12,random-v12}-r{1,2,3,4}.yaml  # Round configs
├── scripts/
│   ├── shepherd_iterative.sh      # Multi-round automation
│   ├── expand_corpus.py           # Adds beliefs between rounds
│   ├── embed_keepalive.py         # Prevents embed server timeout
│   └── slurm/iterative_compass_gpt_oss_120b.sbatch
├── processed_data/
│   ├── compass-gpt-oss-120b-{v12,random-v12}-r{1,2,3,4}/  # Game outputs
│   └── corpus-r{2,3,4}/           # Expanded corpora per round
└── logs/
```

#### Starting the Embedding Server + Keepalive

The embedding server can timeout due to low GPU utilization. The keepalive script sends periodic embedding requests to prevent this.

```bash
# 1. Start embed server (12h allocation)
sbatch .zetteldev/della/scripts/embed_server.sbatch

# 2. Wait for it to start, then launch keepalive
ssh della
cd ~/src/reason_reckon/experiments/15-curvature-as-compass
nohup python scripts/embed_keepalive.py > logs/keepalive.log 2>&1 &

# Check keepalive status
ps aux | grep embed_keepalive
tail -f logs/keepalive.log
```

The keepalive script:
- Finds the embed server node via `squeue`
- Sets up its own SSH tunnel from login node
- Embeds the Paprika corpus every 30 minutes
- Auto-reconnects when embed server restarts

#### Running the Shepherd Script

The shepherd automates multi-round experiments:
1. Waits for current round to complete
2. Runs `expand_corpus.py` to add beliefs from completed games
3. Submits jobs for next round
4. Monitors and resubmits if jobs die

```bash
# Start from round 1
cd /scratch/gpfs/HENDERSON/km5839/reason_reckon/experiments/15-curvature-as-compass
nohup ./scripts/shepherd_iterative.sh > logs/shepherd_iterative.log 2>&1 &

# Resume from a specific round (e.g., after fixing a bug)
nohup ./scripts/shepherd_iterative.sh --start-round 3 >> logs/shepherd_iterative.log 2>&1 &

# Dry run to verify logic
./scripts/shepherd_iterative.sh --dry-run

# Monitor progress
tail -f logs/shepherd_iterative.log
```

#### Manual Corpus Expansion

If you need to create a corpus manually (e.g., to test R3 before R2 completes):

```bash
# Set up SSH tunnel to embed server first
node=$(squeue -u $USER --name=embed-server-$USER --noheader -O NodeList | tr -d " ")
ssh -f -N -L 12851:localhost:12851 -o StrictHostKeyChecking=no $node

# Run expansion
cd /scratch/gpfs/HENDERSON/km5839/reason_reckon/experiments/15-curvature-as-compass
.venv/bin/python scripts/expand_corpus.py \
    --source-corpus ../03-twenty-questions-curvature/processed_data/derived/2-places-world-50 \
    --game-dirs processed_data/compass-gpt-oss-120b-v12-r1/2-places-world-50 \
                processed_data/compass-gpt-oss-120b-random-v12-r1/2-places-world-50 \
    --output-corpus processed_data/corpus-r2 \
    --embed-api-base http://localhost:12851/v1
```

#### Cleaning Stale Lock Files

Parallel workers use `.inprogress` lock files for coordination. Crashed jobs leave stale locks:

```bash
# Find stale locks (>30 min old)
find processed_data/compass-gpt-oss-120b-v12-r3/2-places-world-50 -name "*.inprogress" -mmin +30

# Clean them
find processed_data/compass-gpt-oss-120b-v12-r3/2-places-world-50 -name "*.inprogress" -mmin +30 -delete
```

#### ORC Curvature Parallelization

The `orc_proc` parameter controls parallel edge curvature computation:
- Default: `proc=4` (4 CPU cores per worker)
- Configured in `curvature_compass.py` and passed via config
- With 10 workers × 4 CPUs = 40 CPUs requested in sbatch

To adjust:
1. Edit `iterative_compass_gpt_oss_120b.sbatch`: `--cpus-per-task=40`
2. Edit `curvature_compass.py`: `orc_proc: int = 4`
3. Push changes and pull on Della

#### Monitoring Commands

```bash
# Check all rounds status
for round in 1 2 3 4; do
  for cond in v12 random-v12; do
    dir="processed_data/compass-gpt-oss-120b-${cond}-r${round}/2-places-world-50"
    echo "$cond r$round: $(find $dir -name '*.json' | wc -l)/50"
  done
done

# Check shepherd
ps aux | grep shepherd_iterative
tail -20 logs/shepherd_iterative.log

# Check embed server time remaining
squeue -u $USER --name=embed-server-$USER -o "%j %T %L"

# Check keepalive
tail -10 logs/keepalive.log

# Sync results to local
rsync -avz della:/scratch/gpfs/HENDERSON/km5839/reason_reckon/experiments/15-curvature-as-compass/processed_data/ \
    ~/src/reason_reckon/experiments/15-curvature-as-compass/processed_data/
```

#### Common Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| Embed server timeout | Jobs fail with connection refused | Restart embed server, check keepalive |
| Stale locks blocking progress | Games stuck at same count | Clean locks older than 30 min |
| Corpus path wrong | `FileNotFoundError` for embeddings.npy | Config paths are relative to config/ dir, use `../processed_data/` |
| Shepherd died | No new jobs submitted | Check logs, restart with `--start-round N` |
| Model stuck at same turn | Turn N repeating in logs | May be model loop; will timeout and retry |

## Local vLLM Testing (gpt-oss and others)

### Server Configuration

Local vLLM servers run on the workstation with 2x 4090 GPUs:

- **Main model server**: Port 8013, token `token-local`
- **Embedding server**: Port 8012, token `token-local`

**Environment variables for scripts:**
```bash
OPENAI_API_BASE=http://localhost:8013/v1 \
OPENAI_API_KEY=token-local \
EMBEDDING_API_BASE=http://localhost:8012/v1 \
EMBEDDING_API_KEY=token-local \
uv run python scripts/run_compass.py --config config/compass-gpt-oss-20b.yaml
```

### gpt-oss Specific Configuration

gpt-oss requires specific vLLM settings:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-20b \
  --port 8013 \
  --max-model-len 32768 \
  --tool-call-parser openai \
  --api-key token-local
```

Key settings:
- `--tool-call-parser openai`: Required for proper tool call formatting (not `hermes` or `seed_oss`)
- `--max-model-len 32768`: Default 8192 causes context truncation mid-game
- Model name: `openai/gpt-oss-20b` → litellm config needs `openai/openai/gpt-oss-20b` (double prefix)

### gpt-oss Tool Name Quirk

gpt-oss sometimes generates malformed tool names with `<|channel|>` suffixes:
```
submit_question<|channel|>commentary  (instead of submit_question)
```

**Fix**: Monkeypatch lisette's tool dispatch in your script:
```python
import re
import lisette.core as lisette_core

def _normalize_tool_name(fn: str) -> str:
    return re.sub(r'<\|channel\|>.*$', '', fn)

_original = lisette_core._lite_call_func
def _patched(tc, tool_schemas, ns, raise_on_err=True):
    tc.function.name = _normalize_tool_name(tc.function.name)
    return _original(tc, tool_schemas, ns, raise_on_err)
lisette_core._lite_call_func = _patched
```

Without this, model receives "Tool not defined" errors, gets confused, and abandons tool use entirely.

### Error Handling for Tool Calls

Add robust error handling for malformed tool calls:
```python
consecutive_malformed = 0
try:
    chat("Please continue", max_steps=1)
    consecutive_malformed = 0  # Reset on success
except litellm.BadRequestError as e:
    consecutive_malformed += 1
    # Don't send error message to model - it makes things worse
    if consecutive_malformed >= 10:
        break  # Abort to avoid infinite loop
```

**Important**: Don't send "Your response could not be parsed" messages - this confuses gpt-oss and causes it to abandon the game or reset.

### Prompt Engineering for 20Q-style Games

Games require explicit end-game guidance or models will guess prematurely and "reset":

```
## Winning the Game

The target is often surprisingly specific - not "Paris" but "the Catacombs of Paris."

**Critical**: The game continues until you receive "Game Complete." Until then:
- Never declare a "final answer" - there is no such move
- Never try to end or reset the game
- "Please continue the game" = keep narrowing THIS target
- If your hypothesis is wrong, that's information - pivot and keep asking
```

Without this, models (especially gpt-oss) will:
1. Make premature guesses after ~10 questions
2. Declare "The answer is X!"
3. Interpret "Please continue" as "start a new game"
4. Reset their state to "I have no clues yet"

### Debugging "Forgotten Context"

When a model seems to forget earlier information, check if it's truncation vs behavioral reset:

**Behavioral reset** (more common):
- Search chat for: "new round", "start fresh", "reset", "no clues yet"
- Model explicitly announces starting over
- Check reasoning field for "Did we properly end? We can start again..."

**True context truncation**:
- Model gradually loses early facts without announcing reset
- Check token count vs `max-model-len`
- Estimate: ~4 chars per token, system prompt ~700 tokens

### Server Management

```bash
# Check what's running
curl -s http://localhost:8013/v1/models -H "Authorization: Bearer token-local" | jq .

# Kill gpt-oss server
pkill -f "vllm.*gpt-oss"

# Check GPU usage
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# Monitor server logs
tail -f /tmp/vllm-gpt-oss.log
```

## Data
- The `processed_dir` directories are backed up to Huggingface. This can be managed by the `just hugdata` script.
- `just hugdata status` shows which experiments are unpushed.
- `just hugdata push 02-full-experiment-name` pushes an experiment's `processed_dir` folder.

## JAX/Diffusion Curvature
- Diffusion curvature requires `jax[cuda12]` for GPU acceleration
- Without it, JAX falls back to CPU with warning: "CUDA-enabled jaxlib is not installed"
- Install with: `uv add "jax[cuda12]"`
- Set JAX environment in sbatch scripts:
  ```bash
  JAX_ENV=(CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.75)
  env "${JAX_ENV[@]}" python scripts/run_baseline.py ...
  ```

## Weights & Biases
- `scripts/upload_exp*_to_wandb.py` - Upload experiment results to wandb
- `scripts/create_wandb_report.py` - Programmatic report generation
- Project: `reason-reckon`, Entity: `pumberton`
- Use `REPORT_ID` constant to update existing reports instead of creating new ones
- Per-game results stored in wandb Tables for paired statistical tests
