# Zetteldev

This is a 'Zetteldev' repo for conducting academic research through literate programming. All experimentation and coding must be grounded by markdown design documents supplied by the user. You, programming genie as you are, have the job of being an exceptionally capable (and rather witty) *interpreter* of the user's literate programs, translating the (ideally) highly efficient and expressive combination of English, mathematical notation and pseudocode into robust and interpretable python. 

## Shorthand Commands

- **`rar`** (Reread and Run): Re-read the last active zettel (daily note or experiment zettel), find any new placeholder figures/tables/requests, create scripts to produce them (saved to well-organized python files in appropriate subdirectories), and run them. Always save code to files — never run long computations inline. When a computation is expected to take a while, save useful intermediate results (dataframes, arrays, model outputs) to `processed_data/` so they can be reused by downstream figure scripts without recomputation.

## House Style: Headings and Reports

- **MINI HEADERS**: in long responses and documents, use informal capitalized headings as super-punctuators — each marks the end of one unit of thought, as the comma ends a clause and the period a sentence. Aim for witty headings that reveal the subject of what follows while hinting at the mystery.
- **PREVIOUSLY ON 'INVENTIVE NAME'**: begin each report that follows a long work bout with a recap section titled `PREVIOUSLY ON '<inventive name for the effort>'` — a brief review of the project, the recent work, the current status, and the key definitions/terms in play, so the reader (the author returning after hours away) can re-enter cold.

# Procedure: Phases of an Experiment

**The markdown files which guide and document each experiment are found within the user's Obsidian vault, in ~/Pumberton.** These are standard Obsidian markdown: they contain [[wikilinks]] and YAML frontmatter.

**Most work happens within `experiments/` subfolders.** You will either be asked to work within an existing experiment or create a new one.
   - If in an existing experiment, familiarize yourself with the existing code. Crucially, also ensure you have located and read the 'zettel' in Obsidian. If the user doesn't provide it, consult `design.md` for the path. If the user provides a filename, look in `~/Pumberton/Stream` - this is the default location. 
   - If creating an experiment, run `just create-experiment experiment-name`. This will scaffold numbered marimo notebooks, a main.py, and a Snakefile. For experiments that need cluster training with hyperparameter sweeps, use `just create-experiment experiment-name --hydra` to also scaffold a Hydra config directory and train.py entry point.

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
│   │   ├── design.md (stub pointer to the zettel)
│   │   ├── 01-foundational-marimo-notebook.py (e.g. sets up data)
│   │   ├── 02-method-development.py (e.g. derives the machinery)
│   │   ├── 03-analysis.py (e.g. interactive plots galore)
│   │   ├── main.py
│   │   ├── Snakefile
│   │   ├── scripts/ (python files orchestrated by Snakemake)
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
- `Snakefile`: Defines the experiment's computation DAG (explained below).
- `01-…py`, `02-…py`, `03-…py`: numbered **marimo notebooks** — the computational essays where datasets, methods, and analyses are developed (see "Programs as Essays" below). These replace the old `report.qmd` Quarto flow.
- `main.py`: Main Python script controlling the experiment's large-scale logic.
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

## Programs as Essays: Marimo Notebooks

Each experiment is a conversation between three literate mediums: the *zettel* (design, then results), the *marimo notebooks* (which develop and illustrate the core algorithms, datasets, and machinery), and the *Snakemake scripts* (which perform the experiment at scale). The notebooks are computational essays in Knuth's sense — compiled by the machine, addressed to a human.

Working practices:
- To write in Marimo, refer to your `marimo-pair` skill for the latest API details; in brief, you can connect directly to the kernel of a notebook the author has created, run existing cells, execute 'anonymous' scratchpad code to prototype, and author cells to build the essay. **Always use marimo's "code mode" to edit and create cells; never edit the raw `.py` file** — it is frequently overwritten by the active kernel.
- Marimo demands a functional style: define as objects the primitives considered in the essay, and evolve them through stacks of short, interpretable functions. Heavy computations (trainings, embeddings, analyses up to ~30 min) may occur in the notebook using local resources (beefy CPU, 2× 4090s), but use marimo's cache helpers to prevent needless recomputation.
- Tell, and show: populate the essay with visualizations, mermaid diagrams, math, and raw dataframe views. Unlike the zettel (the author's territory), a beautiful, thoroughly-correct computational essay is *your* responsibility — double-check and red-team it.
- The core pieces of each essay should be importable downstream. A function or class defined in a marimo notebook can be imported by scripts if (1) it is defined in its own cell, with nothing else, and (2) that cell refers only to symbols defined at the *top* of the DAG (e.g. the setup cell). Then:

```python
from zetteldev import notebook_module

bex = notebook_module("02-method-development.py")  # loads a numbered notebook by path
result = bex.core_function(...)                    # an @app.function cell, callable downstream
```

- Publishing figures from within a notebook, without leaving the kernel:

```python
from zetteldev import figpub
figpub.publish(fig, "difficulty_spectrum.png")  # uploads to R2, dedupes, returns the URL
```

  It accepts a matplotlib `Figure`/`Axes`, a `PIL.Image`, a path, or raw bytes; infers the experiment and `figures/` directory from the working directory; and returns an object rendering the figure inline beside its stable URL and a copy-paste `![…](url)` line. Identical bytes are never re-uploaded.
- Marimo quirk: the notebook's kernel changes ID whenever the webpage is refreshed (author views from a new device, connection drop). If the kernel suddenly seems unresponsive, rediscover its new ID.

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
- Always define an `all` rule so the entire experiment can be run with `pixi run snakemake`.
- Pure-CPU prep steps should be marked `localrule: True` so they run on the login node / local machine without a SLURM allocation.
- For cluster pipelines, a Snakemake SLURM *profile* lives at `.zetteldev/snakemake/della/config.yaml` (`executor: slurm`): each GPU rule becomes its own SLURM job while `localrule` steps run on the login node. On della: `export SNAKEMAKE_PROFILE=$REPO/.zetteldev/snakemake/della` then `pixi run snakemake -j8 <target>` from a tmux window. The profile sets the concurrency cap, an NFS `latency-wait`, one auto-resubmit on transient failure, and cheap default resources for un-annotated CPU rules.
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
- `zetteldev.figpub.publish(fig, "name.png")` — from inside a marimo notebook (preferred; see "Programs as Essays")
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

1. *Is it doing what the user's design spec wants?* While writing scripts, write corresponding tests in the `./tests` folder, using Pytest. Execute these with `just test` (runs all tests in the project) or `pixi run pytest path/to/test_file.py` for a specific file. Because there are multiple experiment directories, include the current experiment name in any tests you write to prevent collisions.
2. *Does it run without errors?* In all scripts, respect a `test_run` parameter (set globally in the Snakefile) which performs only the bare minimum computation to use all bits of the code. For example, process a tiny subset of the input data; do only 2 epochs of training; use only 2 rounds of monte-carlo sampling. After implementing an experiment, set the `test_run` to true and run snakemake. Ensure the full pipeline works.

---

# The Driver: Long Marches

When an experiment reaches the "Marches" — code written, awaiting results from jobs — the author will ask you to employ a *driver*: a `/loop` cron job that wakes you every third hour to keep things progressing. Anchor each driver in the experiment's zettel; the author adds a task list at the bottom of the zettel (experiments to run, results to gather).

On each driver wake:
1. Re-read the entire zettel — context and tasks may have changed since the last check-in.
2. Identify the unblocked tasks.
3. Complete as many as you can. Keep *one thing* in focus at a time; when it blocks (waiting on a run, or on user input), move to the next. Tasks are ordered roughly by priority; adjust from your knowledge of the details.
4. Report progress tersely (counts and a rate estimate); if jobs left the queue, read their logs for the cause; if they completed, clear stale `.inprogress` locks and resubmit the remainder.

Create monitoring loops yourself whenever the need arises — immediately after submitting jobs, and again after any mop-up resubmission. Favour many short jobs over one long one (conservative 3–6h limits earn queue priority), and design sweeps to **resume** so a timeout costs nothing.

# Conventions

## Environment & Package Management
This project uses **pixi** for Python package management (config in `pyproject.toml` under `[tool.pixi.*]`) and **just** as a task runner.

### pixi basics
- `pixi install` - Install the environment from pyproject.toml + pixi.lock
- `pixi run python script.py` - Run a script in the environment (or call `.pixi/envs/default/bin/python` directly — more reliable for detached/background processes, where `pixi run` can hang)
- `pixi add package` - Add a conda dependency; `pixi add --pypi package` for PyPI
- **Gotcha**: after editing dependencies in `pyproject.toml` by hand, run `pixi lock` explicitly — `pixi install` alone may silently skip re-solving.
- The `zetteldev` package (`.zetteldev/`, editable install) provides `figpub`, `notebook_module`, and cluster helpers.

Dependencies are defined in `pyproject.toml`. If you need a package not listed, please ask. (A `uv.lock` also exists for pip-compatible tooling, but pixi is the source of truth.)

### just task runner
Common tasks are defined in `justfile`. Run `just` to see all available commands:
- `just test` - Run pytest
- `just create-experiment` - Create a new experiment (scaffolds marimo notebooks)
- `pixi run marimo edit <notebook.py>` - Open/serve a marimo notebook (the author usually runs the kernel; connect via the `marimo-pair` skill)

See the full list with `just --list`.

## Code Style Guidelines

- **Python Version**: 3.11+
- **Imports**: Standard library first, third-party packages, local modules
- **Type Annotations**: Use typing for function parameters and returns
- **Documentation**: Google-style docstrings with params and returns
- **Error Handling**: Specific exceptions with proper logging
  - Don't allow things to silently fail. If something critical to the experiment (i.e. a specified model, a necessary API key) is missing, the researcher needs to be informed immediately.

# Tips 

## Della Eval Army (primary cluster pattern for this repo)

Benchmark evaluations parallelize via the *eval army*: generic worker-packed SLURM jobs over shard-resume scripts. All benchmark scripts share the `(--worker-id, --num-workers, --device)` + per-worker CSV shard pattern, so one sbatch serves them all.

- `experiments/11-successor-curvatures/slurm/eval_army.sbatch` — packs `WORKERS_PER_GPU` (default 8) workers per GPU (our nets/OT problems are small; a lone worker wastes ~90% of an H100). Parameterized by env: `SCRIPT`, `SUBCMD`, `JOB_INDEX`, `NUM_JOBS`, `WORKDIR` (run scripts from any experiment dir), `SCRIPT_ARGS`.
- `submit_army.sh <script> <subcmd> <jobs> <gpus> <partition> [extra args]` — tiles the worker range across jobs; `pull_shards.sh "<glob>"` — rsync shards back for local merge.
- Typical scale: 4 jobs × 4 H100 × 8 workers = 128 slots; a full-battery FB-heavy run lands in 15–20 min (vs ~3 h local). Della↔local replication verified exact (corr ≥ 0.9998 on identical instances).
- **Shard-resume discipline**: workers skip instances already present in their own shard, so stragglers/mop-ups are cheap: `scancel` the job, resubmit its `JOB_INDEX` (optionally `--exclude=<node>`); only missing instances recompute. Never commit per-worker shards to git (cross-machine collisions) — merged CSVs only; `experiments/*/processed_data/*_w[0-9]*.csv` is gitignored.
- **Sync discipline**: code moves by git (`git push` locally; on della `git stash -q` FIRST, then `git pull`, then **verify HEAD** — pulls fail silently on unstaged changes); data moves by rsync (`processed_data` is not tracked). The fixed-seed battery joblib must be rsynced, not regenerated.
- Watch for wall-limit races: if a worker's remaining instances can't finish inside the job's time limit, cancel and mop up with finer sharding rather than waiting for the clock.

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
- Install with: `pixi add --pypi "jax[cuda12]"`
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
