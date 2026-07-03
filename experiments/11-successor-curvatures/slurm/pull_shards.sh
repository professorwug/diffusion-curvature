#!/bin/bash
# Pull result shards back from della (run locally from the experiment dir).
# Usage: ./slurm/pull_shards.sh <shard-glob>   e.g. "diffusing_fraction_w*.csv"
set -euo pipefail
cd "$(dirname "$0")/.."
rsync -avz "della:~/src/diffusion-curvature/experiments/11-successor-curvatures/processed_data/${1}" processed_data/
