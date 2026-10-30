#!/usr/bin/env bash
# Modified for portable private snapshot; see docs/snapshot-changes.md.
set -euo pipefail
motion_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
motion_python="${MOTION_TSNE_PYTHON:-python}"
if ! command -v "$motion_python" >/dev/null 2>&1; then
  printf 'Python not found: %s. Set MOTION_TSNE_PYTHON to your existing environment.\n' "$motion_python" >&2
  exit 1
fi
exec "$motion_python" "$motion_project_dir/run.py" "$@"
