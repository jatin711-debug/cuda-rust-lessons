#!/usr/bin/env bash
# Verify configurable kernels immediately below, at, and above a 128-wide tile.
set -euo pipefail
task_lab_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
selection="${1:-all}"
case "$selection" in
  tile|oxide|all) ;;
  *) printf 'Usage: bash scripts/verify_gpu_edges.sh [tile|oxide|all]\n' >&2; exit 2 ;;
esac
for length in 1 127 128 129 257; do
  for lesson in saxpy pipeline; do
    bash "$task_lab_root/scripts/run_gpu.sh" "$selection" "$lesson" "$length" 2
  done
done
printf '\nAll selected SAXPY/pipeline boundary cases passed.\n'
