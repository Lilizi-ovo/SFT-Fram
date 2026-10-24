#!/usr/bin/env bash
set -euo pipefail
FRAM_LAUNCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${FRAM_PYTHON:-}" ]]; then
    FRAM_LAUNCH_PYTHON="$FRAM_PYTHON"
elif [[ -x "$FRAM_LAUNCH_ROOT/.venv/bin/python" ]]; then
    FRAM_LAUNCH_PYTHON="$FRAM_LAUNCH_ROOT/.venv/bin/python"
else
    FRAM_LAUNCH_PYTHON=python3
fi
exec "$FRAM_LAUNCH_PYTHON" "$FRAM_LAUNCH_ROOT/main.py" "$@"
