#!/usr/bin/env bash
# Regression check: does the current OASIS code reproduce the reference?
#
# Usage:
#   bash validation/run_regression.sh <small|dense> <work_dir> [--tiled]
#
#   small   ~4.1M particles, about 1 minute per run
#   dense   ~59M particles, several minutes per run and several GB of disk
#   work_dir   folder for the box, the exported reference code and all outputs
#              (created if needed; reused boxes are not regenerated)
#   --tiled    additionally run the current code through the tiled pipeline
#              (2 tiles per side) and require it to match the normal run
#
# Checks:
#   1. reference vs current with seed_tie_break='load_order' (the reference's
#      order for seeds with equal M200b): must PASS in --mode reference.
#   2. (--tiled) current mini-box run vs current tiled run, both with the
#      default seed_tie_break='halo_id': must PASS in --mode tiled.
#   3. informational: how many haloes the default tie-break changes compared
#      with the reference order (load_order vs halo_id). Never fails.
#
# Environment: THREADS (default 8) sets the number of worker processes.
# Exit code: 0 if every comparison passes, 1 otherwise.
set -euo pipefail

if [[ $# -lt 2 || ( "$1" != "small" && "$1" != "dense" ) ]]; then
    echo "usage: bash $0 <small|dense> <work_dir> [--tiled]" >&2
    exit 2
fi
BOX_NAME="$1"
WORK="$(mkdir -p "$2" && cd "$2" && pwd)"
TILED="${3:-}"
THREADS="${THREADS:-8}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"

echo "== repository:   $REPO ($(git -C "$REPO" rev-parse --short HEAD))"
echo "== work folder:  $WORK"

# 1. Reference code, exported from the oasis-reference tag (no second clone).
if ! git -C "$REPO" rev-parse -q --verify "refs/tags/oasis-reference" >/dev/null; then
    echo "error: tag 'oasis-reference' not found in $REPO" >&2
    exit 2
fi
REF_PKG="$WORK/reference_code"
rm -rf "$REF_PKG"
mkdir -p "$REF_PKG"
git -C "$REPO" archive oasis-reference oasis | tar -x -C "$REF_PKG"
echo "== reference:    oasis-reference ($(git -C "$REPO" rev-parse --short 'oasis-reference^{commit}'))"

# 2. The regression box (deterministic, so it is only generated once).
BOX="$WORK/${BOX_NAME}_box.npz"
if [[ ! -f "$BOX" ]]; then
    echo "== generating $BOX_NAME box"
    python "$HERE/make_regression_box.py" "$BOX_NAME" "$BOX"
fi

# 3. Runs. Only these output folders inside work_dir are ever deleted.
RUNS="$WORK/runs_$BOX_NAME"
mkdir -p "$RUNS"
run() {   # run <name> <package_dir> [extra args]
    local name="$1" pkg="$2"; shift 2
    rm -rf "${RUNS:?}/$name"
    echo "== running $name"
    python "$HERE/run_pipeline.py" "$pkg" "$BOX" "$RUNS/$name" --threads "$THREADS" "$@" \
        > "$RUNS/$name.log" 2>&1 || { echo "run $name failed, see $RUNS/$name.log" >&2; exit 1; }
    grep -E '"(preprocess|calibration_data|build_tiles|catalogue_total)"' "$RUNS/$name.log" || true
}
run reference "$REF_PKG"
run current_load_order "$REPO" --seed-tie-break load_order
run current "$REPO"
[[ "$TILED" == "--tiled" ]] && run current_tiled "$REPO" --tiles 2

# 4. Comparisons.
status=0
echo; echo "== 1. reference vs current (seed_tie_break=load_order)"
python "$HERE/compare_runs.py" "$RUNS/reference" "$RUNS/current_load_order" --mode reference || status=1
if [[ "$TILED" == "--tiled" ]]; then
    echo; echo "== 2. current mini-boxes vs current tiled (seed_tie_break=halo_id)"
    python "$HERE/compare_runs.py" "$RUNS/current" "$RUNS/current_tiled" --mode tiled || status=1
fi
echo; echo "== 3. informational: effect of the default tie-break (load_order -> halo_id)"
python "$HERE/compare_runs.py" "$RUNS/current_load_order" "$RUNS/current" --count-only

echo
if [[ $status -eq 0 ]]; then echo "REGRESSION: PASS"; else echo "REGRESSION: FAIL"; fi
exit $status
