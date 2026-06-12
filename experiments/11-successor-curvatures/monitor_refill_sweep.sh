#!/usr/bin/env bash
# Monitor for the refill sweep. Polls every POLL_SEC seconds.
#
# Behaviour each iteration:
#   1. If both final figures already exist → DONE, exit 0.
#   2. If a snakemake orchestrator is alive → sleep and check again.
#   3. Otherwise (orchestrator dead, figures missing) → relaunch snakemake.
#      Snakemake's --rerun-incomplete + --keep-incomplete combo will pick
#      up partial cell CSVs left by previous SLURM cells and resume them
#      via the script's per-row resume logic.
#
# Stops after MAX_ATTEMPTS relaunches.

set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs

MAX_ATTEMPTS=${MAX_ATTEMPTS:-6}
POLL_SEC=${POLL_SEC:-900}  # 15 min

PARTITION=${SLURM_PARTITION:-cpu}
ACCOUNT=${SLURM_ACCOUNT:-henderson}
CPUS=${CPUS_PER_TASK:-8}
MEM=${MEM_MB:-64000}
RUNTIME=${RUNTIME_MIN:-480}
MAX_JOBS=${MAX_JOBS:-40}

FINAL_TARGETS=(
    figures/v2/tau_colosseum_pearson_compact.png
    figures/v2/colosseum_pearson_compact_v3.png
)

is_done() {
    for t in "${FINAL_TARGETS[@]}"; do
        [[ -f "$t" ]] || return 1
    done
    return 0
}

orchestrator_alive() {
    pgrep -u "$USER" -af "snakemake.*tau_colosseum_pearson_compact" 2>/dev/null \
        | grep -v "monitor_refill_sweep" | grep -v "pgrep" > /dev/null
}

cell_status() {
    local tau iid
    tau=$(ls processed_data/tau_cells 2>/dev/null | wc -l)
    iid=$(ls processed_data/iid_cells 2>/dev/null | wc -l)
    echo "tau=$tau/20 iid=$iid/20"
}

run_snakemake() {
    pixi run snakemake --unlock 2>&1 | tail -2 || true
    pixi run snakemake \
        --executor slurm \
        -j "$MAX_JOBS" \
        --default-resources \
            slurm_partition="$PARTITION" \
            mem_mb="$MEM" \
            runtime="$RUNTIME" \
            cpus_per_task="$CPUS" \
            slurm_account="$ACCOUNT" \
        --config \
            cell_mem_mb="$MEM" \
            cell_runtime="$RUNTIME" \
            cell_cpus="$CPUS" \
        --keep-going \
        --keep-incomplete \
        --rerun-incomplete \
        figures/v2/tau_colosseum_pearson_compact.png \
        figures/v2/colosseum_pearson_compact_v3.png
}

attempt=0
echo "[$(date)] monitor starting (max attempts: $MAX_ATTEMPTS, poll: ${POLL_SEC}s)"
while (( attempt < MAX_ATTEMPTS )); do
    if is_done; then
        echo "[$(date)] both final figures present — DONE."
        exit 0
    fi

    if orchestrator_alive; then
        echo "[$(date)] orchestrator alive; cells: $(cell_status); sleeping ${POLL_SEC}s"
        sleep "$POLL_SEC"
        continue
    fi

    attempt=$((attempt + 1))
    echo "[$(date)] attempt $attempt/$MAX_ATTEMPTS — orchestrator dead, figures missing."
    echo "             current cell counts: $(cell_status)"
    echo "             relaunching snakemake (resume mode)"
    run_snakemake || echo "[$(date)] snakemake exited non-zero — will poll again"
    sleep 60
done

if is_done; then
    echo "[$(date)] FINAL: DONE after $attempt relaunches."
else
    echo "[$(date)] FINAL: gave up after $MAX_ATTEMPTS attempts; figures missing."
    echo "             cell counts: $(cell_status)"
fi
