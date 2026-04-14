#!/usr/bin/env python3
"""
Scaffold a Zetteldev experiment folder with:
  • 16‑char secret token in .zetteldev
  • report.qmd for PDF rendering
  • Snakefile renders report to PDF
  • scratchpad.ipynb Jupyter notebook pre‑wired for live exploration

Usage:
  python create_experiment.py             # Interactive mode
  python create_experiment.py <exp_name>  # Create with specific name

Reads [tool.zetteldev] base_url from pyproject.toml.
"""
import json, secrets, string, sys
import re
from pathlib import Path

try:
    import tomllib            # Python ≥3.11
except ModuleNotFoundError:   # pragma: no cover – for ≤3.10
    import tomli as tomllib

import questionary            # interactive prompt

# ---------------------------------------------------------------------------- #
# helper functions
# ---------------------------------------------------------------------------- #
TOKEN_ALPHABET = string.ascii_lowercase + string.digits

def make_token(n: int = 16) -> str:
    """Return an n‑char random slug suitable for URLs."""
    return "".join(secrets.choice(TOKEN_ALPHABET) for _ in range(n))

def get_base_url() -> str:
    """Read project‑wide base_url from pyproject.toml or fall back."""
    try:
        data = tomllib.loads(Path("pyproject.toml").read_text())
        return data["tool"]["zetteldev"]["base_url"].rstrip("/")
    except Exception:
        return "http://localhost:8000"

def get_existing_experiment_slugs() -> set:
    """Get slugified names (without number prefix) from existing experiment folders."""
    base = Path("experiments")
    if not base.exists():
        return set()

    slugs = set()
    for folder in base.iterdir():
        if folder.is_dir():
            # Strip leading number prefix (e.g., "01-foo" -> "foo")
            match = re.match(r"^\d+-(.+)$", folder.name)
            if match:
                slugs.add(match.group(1))
            else:
                slugs.add(folder.name)
    return slugs


def get_next_experiment_number() -> int:
    """Get the next sequential experiment number."""
    base = Path("experiments")
    if not base.exists():
        return 1

    max_num = 0
    for folder in base.iterdir():
        if folder.is_dir():
            match = re.match(r"^(\d+)-", folder.name)
            if match:
                max_num = max(max_num, int(match.group(1)))
    return max_num + 1

def slugify_title(title: str, max_length: int = 50) -> str:
    """Convert title to a valid folder name."""
    # Remove special characters and convert to lowercase
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    # Replace spaces with hyphens
    slug = re.sub(r"[\s_]+", "-", slug)
    # Remove leading/trailing hyphens
    slug = slug.strip("-")
    # Truncate if too long
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")
    return slug

# ---------------------------------------------------------------------------- #
# templates – double braces → single brace in output
# ---------------------------------------------------------------------------- #

SNAKEMAKE_TEMPLATE = """\
onsuccess:
    shell("uv run python ../../.zetteldev/publish_figure.py --sync {exp}")

rule all:
    input:
        "report.pdf"

rule run_main:
    input:
        data = "../../data/foo.arrow" # placeholder
    output:
        # main.py outputs
    script:
        "main.py"

rule render_report:
    input:
        "report.qmd"
    output:
        "report.pdf"
    shell:
        "uv run quarto render report.qmd --to pdf"
"""

HYDRA_CONFIG_TEMPLATE = """\
defaults:
  - _self_

# Run identity
name: unnamed

# Add your experiment parameters here
# param1: value1
# param2: value2

# Hydra settings
hydra:
  job:
    chdir: false
  run:
    dir: .
  sweep:
    dir: processed_data/sweeps/${{now:%Y-%m-%d}}
    subdir: ${{name}}
"""

HYDRA_LAUNCHER_TEMPLATE = """\
# @package hydra.launcher
# Symlinked from .zetteldev/hydra/launcher/ — edit there to change defaults.
"""

HYDRA_TRAIN_TEMPLATE = """\
\"\"\"Training entry point — Hydra config + optional submitit SLURM submission.

Usage:
    python train.py                             # local run
    python train.py +run=my_preset              # with run preset
    python train.py -m hydra/launcher=pli_c     # submit to SLURM
    python train.py -m param=a,b,c hydra/launcher=pli_c  # sweep
    python train.py --cfg job                   # print resolved config
\"\"\"

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # TODO: Add your training logic here


if __name__ == "__main__":
    main()
"""

REPORT_QMD_TEMPLATE = """\
---
title: "{exp}"
format:
  pdf:
    toc: true
    toc-depth: 3
    number-sections: true
    colorlinks: true
execute:
  echo: false
  warning: false
---

# {exp}

> *Dataset*: Describe what data you used and how?
> *Metrics*: Which metrics did you use and how?
> *Hypothesis*: ...

## Setup

```{{python}}
print("Hello from {exp}")
```
"""

MAIN_PY_TEMPLATE = '''"""
Main module for the {exp} experiment.

This module contains the core functionality for the experiment.
"""
from snakemake.script import snakemake # direct access to Snakefile variables
data_input = snakemake.input.data

print("Running {exp} …")
'''

TEST_BASIC_TEMPLATE = """import os
import sys
import pytest

# Add the parent directory to the path so we can import the main module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import from the current experiment's main.py
from main import *

def test_{exp_underscore}_sanity():
    \"\"\"Basic sanity check for the {exp} experiment.\"\"\"
    assert True, "Basic test for {exp}"
"""

# ---------------------------------------------------------------------------- #
# notebook creation helpers
# ---------------------------------------------------------------------------- #

NOTEBOOK_PREAMBLE = """%load_ext autoreload
%autoreload 2

import sys
import importlib
"""

def create_scratchpad(exp: str, exp_underscore: str, path: Path) -> None:
    """Write a minimal Jupyter notebook with autoreload and imports."""
    # Format the preamble with the experiment name and underscored version
    formatted_preamble = NOTEBOOK_PREAMBLE.format(exp=exp, exp_underscore=exp_underscore)

    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [f"# Scratchpad for **{exp}** Experiment"]},
            {"cell_type": "code", "metadata": {}, "source": formatted_preamble.splitlines(True), "outputs": [], "execution_count": None},
            {"cell_type": "code", "metadata": {}, "source": "", "outputs": [], "execution_count": None},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython", "version": sys.version.split()[0]},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=2))

# ---------------------------------------------------------------------------- #

def scaffold_experiment(exp: str, hydra: bool = False) -> None:
    base = Path("experiments")
    folder = base / exp
    if folder.exists():
        sys.exit(f"⚠️  Experiment {exp!r} already exists.")

    # create folders
    folder.mkdir(parents=True)
    for sub in ("processed_data", "figures", "tests"):
        (folder / sub).mkdir()

    token = make_token()
    url   = f"{get_base_url()}/experiments/{exp}-{token}"

    # metadata
    (folder / ".zetteldev").write_text(f"token={token}\nurl={url}\n", encoding="utf-8")

    # write core files
    exp_underscore = exp.replace("-", "_")

    (folder / "main.py").write_text(MAIN_PY_TEMPLATE.format(exp=exp))
    (folder / "report.qmd").write_text(REPORT_QMD_TEMPLATE.format(exp=exp))
    (folder / "Snakefile").write_text(SNAKEMAKE_TEMPLATE.format(exp=exp, token=token))
    (folder / "tests" / f"test_{exp_underscore}.py").write_text(TEST_BASIC_TEMPLATE.format(exp=exp, exp_underscore=exp_underscore))

    # scratchpad notebook with experiment name in filename
    create_scratchpad(exp, exp_underscore, folder / f"scratchpad_{exp_underscore}.ipynb")

    # Hydra + submitit scaffold for cluster experiments
    if hydra:
        conf_dir = folder / "conf"
        conf_dir.mkdir()
        (conf_dir / "run").mkdir()
        (conf_dir / "config.yaml").write_text(HYDRA_CONFIG_TEMPLATE)
        (folder / "train.py").write_text(HYDRA_TRAIN_TEMPLATE)

        # Symlink shared launcher configs from .zetteldev/hydra/
        zetteldev_hydra = Path("../../.zetteldev/hydra")
        hydra_dir = conf_dir / "hydra"
        hydra_dir.mkdir()
        (hydra_dir / "launcher").symlink_to(zetteldev_hydra / "launcher")

        print(f"🔧  Hydra scaffold added: conf/, train.py, launcher symlink")

    print(f"✅  Experiment {exp!r} scaffolded at {folder}")
    print(f"🔗  Canonical URL: {url}")

# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Parse --hydra flag
    use_hydra = "--hydra" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--hydra"]

    if args:
        # Treat argument as an experiment name
        slug = slugify_title(args[0])
        exp_num = get_next_experiment_number()
        name = f"{exp_num:02d}-{slug}"
        print(f"📋 Creating experiment '{name}'" + (" (with Hydra)" if use_hydra else ""))
        scaffold_experiment(name, hydra=use_hydra)
    else:
        # Interactive mode - prompt for name
        if not sys.stdin.isatty():
            sys.exit("⚠️  No experiment name provided. Usage: python create_experiment.py <name> [--hydra]")

        try:
            slug = questionary.text("New experiment name:").ask()
            if not slug:
                sys.exit("No name provided.")
            slug = slugify_title(slug)
        except (KeyboardInterrupt, EOFError):
            sys.exit("\n⚠️  Cancelled.")
        except Exception as e:
            sys.exit(f"⚠️  Error getting experiment name: {e}")

        # Create folder name with sequential number prefix
        exp_num = get_next_experiment_number()
        name = f"{exp_num:02d}-{slug}"
        scaffold_experiment(name, hydra=use_hydra)
