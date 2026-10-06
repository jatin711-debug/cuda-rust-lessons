#!/usr/bin/env bash
set -euo pipefail
task_lab_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
for lesson in warp-reduction prefix-scan; do
  for length in 1 31 32 33 127 128 129 16384 16385 1048576; do
    bash "$task_lab_root/scripts/run_gpu.sh" oxide "$lesson" "$length" 2
  done
done
for batch in 1 15 16 17 33 1024; do
  bash "$task_lab_root/scripts/run_gpu.sh" oxide gpu-classifier "$batch" 2
done
printf '\nAll 26 deep-core GPU cases passed.\n'
