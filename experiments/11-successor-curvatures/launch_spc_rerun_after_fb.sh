#!/usr/bin/env bash
# Watcher: wait for the in-flight FB-rerun snakemake to finish, then trigger
# an SPC re-run alongside (lifts the d≤3 cap so SPC fills the full dim grid).
#
# This script polls the local file system + process table on Della; once it
# detects that the FB rerun has produced both merged-CSV outputs and the
# orchestrator process is gone, it strips SPC rows from per-cell CSVs and
# resubmits the Snakemake DAG with --forcerun for the cell rules.
#
# Run with nohup so it survives ssh logout:
#     nohup ./launch_spc_rerun_after_fb.sh > logs/spc_rerun_watch.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs

echo "[$(date)] watcher started — waiting for FB rerun to finish."
while true; do
    # The FB rerun is "done" when both merged CSVs exist AND no snakemake
    # orchestrator is running. The merged CSVs are deleted at the start of
    # each rerun and rebuilt by the final merge rule, so their presence is
    # a strong signal.
    if [[ -f processed_data/tau_metrics.csv && -f processed_data/iid_metrics_v3.csv ]]; then
        if ! pgrep -u "$USER" -af "snakemake.*tau_colosseum_pearson_compact|snakemake.*colosseum_pearson_compact_v3" > /dev/null; then
            echo "[$(date)] FB rerun appears finished."
            break
        fi
    fi
    sleep 60
done

echo "[$(date)] stripping SPC rows from per-cell CSVs …"
pixi run python strip_successor_rows.py \
    --methods spc_hop spc_diffusion_t5 \
    processed_data/iid_cells/*.csv \
    processed_data/tau_cells/*.csv

echo "[$(date)] removing merged outputs so they rebuild from updated cells …"
rm -f \
    processed_data/tau_metrics.csv \
    processed_data/iid_metrics_v3.csv \
    processed_data/v2/tau_colosseum_pearson_summary.csv \
    processed_data/v2/colosseum_pearson_summary_v3.csv \
    figures/v2/tau_colosseum_pearson_compact.png \
    figures/v2/tau_colosseum_pearson_compact.svg \
    figures/v2/tau_colosseum_pearson_compact.pdf \
    figures/v2/colosseum_pearson_compact_v3.png \
    figures/v2/colosseum_pearson_compact_v3.svg \
    figures/v2/colosseum_pearson_compact_v3.pdf

pixi run snakemake --unlock 2>&1 | tail -2 || true

echo "[$(date)] resubmitting cells via Snakemake (forced) …"
exec pixi run snakemake \
    --executor slurm \
    -j 40 \
    --default-resources slurm_partition=cpu mem_mb=64000 runtime=240 cpus_per_task=8 slurm_account=henderson \
    --config cell_mem_mb=64000 cell_runtime=240 cell_cpus=8 \
    --keep-going \
    --rerun-incomplete \
    --forcerun tau_cell iid_cell \
    figures/v2/tau_colosseum_pearson_compact.png \
    figures/v2/colosseum_pearson_compact_v3.png
