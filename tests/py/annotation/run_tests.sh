#!/usr/bin/env bash
# Run the full annotation test suite.
#
# Pass 1 (main):    all tests; conftest pins to Blackwell GPU, skips requires_pre_bw.
# Pass 2 (pre-BW):  requires_pre_bw tests; conftest pins to first pre-Blackwell GPU.
#
# Both passes run in parallel — conftest.py selects the GPU for each process.
#
# Usage:
#   bash tests/py/annotation/run_tests.sh [extra pytest args]
#
# Examples:
#   bash tests/py/annotation/run_tests.sh
#   bash tests/py/annotation/run_tests.sh --tb=long
#   bash tests/py/annotation/run_tests.sh -k test_add_one

set -euo pipefail

# Ensure optional test dependencies are installed.
# cuda-tile[tileiras] provides the tileiras AOT compiler for CuTile plugin tests.
# nvidia-cutlass-dsl provides the CuTeDSL Python frontend for CuTeDSL plugin tests.
_missing=()
python -c "import cuda.tile" 2>/dev/null || _missing+=("cuda-tile[tileiras]")
python -c "import cutlass.cute" 2>/dev/null || _missing+=("nvidia-cutlass-dsl")
if [ ${#_missing[@]} -gt 0 ]; then
    echo "Installing missing test deps: ${_missing[*]}"
    pip install "${_missing[@]}" --quiet
fi
unset _missing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$SCRIPT_DIR"
EXTRA_ARGS=("$@")

OUTDIR="/tmp/tta_tests_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
LOG_MAIN="$OUTDIR/pass1_main.log"
LOG_PRE_BW="$OUTDIR/pass2_pre_bw.log"
LOG_SUMMARY="$OUTDIR/summary.txt"

echo "Output directory: $OUTDIR"
echo "Running both passes in parallel (different GPUs)..."

rc1=0
rc2=0

python -m pytest \
    "$TEST_DIR" \
    -n 12 --tb=short -v \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_MAIN" &
PID1=$!

_TTA_PRE_BW_SUBPROCESS=1 \
python -m pytest \
    -m requires_pre_bw \
    "$TEST_DIR" \
    -n 12 --tb=short -v \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_PRE_BW" &
PID2=$!

wait $PID1 || rc1=$?
wait $PID2 || rc2=$?

overall_rc=$(( rc1 | rc2 ))

{
    echo "=== Pass 1 summary (main / Blackwell) ==="
    grep -E "passed|failed|error" "$LOG_MAIN" | tail -3
    echo ""
    echo "=== Pass 2 summary (pre-BW) ==="
    grep -E "passed|failed|error" "$LOG_PRE_BW" | tail -3
    echo ""
    if [ "$overall_rc" -eq 0 ]; then
        echo "OVERALL: PASSED"
    else
        echo "OVERALL: FAILED (pass1_rc=$rc1, pass2_rc=$rc2)"
    fi
} | tee "$LOG_SUMMARY"

echo ""
echo "Logs saved to: $OUTDIR"
exit "$overall_rc"
