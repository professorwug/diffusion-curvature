#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import subprocess
import questionary

def list_experiments() -> list[str]:
    """Return a list of subdirectories in 'experiments'."""
    base_dir = Path("experiments")
    if not base_dir.exists():
        return []
    return sorted([
        d.name for d in base_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

def main():
    experiments = list_experiments()
    if not experiments:
        print("No experiments found in 'experiments' directory.")
        sys.exit(0)

    # Prompt user with autocomplete
    experiment_name = questionary.autocomplete(
        "Select an experiment to run:",
        choices=experiments,
        validate=lambda text: text in experiments or "Please choose a valid experiment."
    ).ask()

    if not experiment_name:
        print("No experiment selected. Exiting.")
        sys.exit(0)

    exp_dir = Path("experiments") / experiment_name
    if not exp_dir.exists():
        print(f"Experiment folder '{experiment_name}' does not exist.")
        sys.exit(1)

    # Change directory into the experiment folder
    os.chdir(exp_dir)

    # Run Snakemake
    print(f"Running Snakemake in '{exp_dir}'...")
    result = subprocess.run(["snakemake"], check=False)

    if result.returncode == 0:
        print("Snakemake completed successfully.")
    else:
        print("Snakemake encountered an error.")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
