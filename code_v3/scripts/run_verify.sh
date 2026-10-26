#!/usr/bin/env bash
# Re-run and verify every reported PEng number (analysis/verify_all.py). Waits for WAIT_PID first.
cd "$(dirname "$0")/.."
if [ -n "${WAIT_PID:-}" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done; fi
python -m analysis.verify_all --run --compare --jobs 4
echo VERIFY_DONE
