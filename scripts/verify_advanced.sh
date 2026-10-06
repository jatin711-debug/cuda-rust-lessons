#!/usr/bin/env bash
set -euo pipefail
task_lab_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
for length in 1 127 128 129 16384 16385 1048576; do
  bash "$task_lab_root/scripts/run_gpu.sh" oxide device-reduction "$length" 2
done
for shape in '1 1' '1 33' '33 1' '31 33' '32 32' '33 31' '35 19' '1024 1024'; do
  read -r rows cols <<< "$shape"
  bash "$task_lab_root/scripts/run_gpu.sh" oxide tiled-transpose "$rows" "$cols" 2
done
for length in 1 129 257 1048576; do
  bash "$task_lab_root/scripts/run_gpu.sh" oxide saxpy-events "$length" 2
done
printf '\nAll 19 advanced GPU cases passed.\n'
