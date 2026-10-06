#!/usr/bin/env bash
set -euo pipefail
task_lab_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
selection="${1:-all}"
lesson_selection="${2:-all}"
if [[ "$selection" == --help || "$selection" == -h ]]; then
  printf 'Usage: bash scripts/run_gpu.sh [tile|oxide|all] [all|lesson] [arguments...]\n'
  printf 'Lessons: easy-vector-add saxpy block-reduction medium-softmax transpose hard-matmul pipeline\n'
  printf 'Advanced oxide lessons: device-reduction tiled-transpose saxpy-events\n'
  printf 'SAXPY/pipeline/device-reduction/saxpy-events: [length] [repeats]. Tiled transpose: [rows] [cols] [repeats].\n'
  exit 0
fi
case "$selection" in
  tile|oxide|all) ;;
  *) printf 'Usage: bash scripts/run_gpu.sh [tile|oxide|all] [all|lesson] [arguments...]\n' >&2; exit 2 ;;
esac
lessons=(easy-vector-add saxpy block-reduction medium-softmax transpose hard-matmul pipeline)
oxide_lessons=("${lessons[@]}" device-reduction tiled-transpose saxpy-events)
case "$lesson_selection" in
  all) ;;
  easy-vector-add|saxpy|block-reduction|medium-softmax|transpose|hard-matmul|pipeline)
    lessons=("$lesson_selection"); oxide_lessons=("$lesson_selection") ;;
  device-reduction|tiled-transpose|saxpy-events)
    lessons=(); oxide_lessons=("$lesson_selection")
    if [[ "$selection" == tile ]]; then
      printf 'Advanced lesson %s is implemented in oxide. Select oxide or all.\n' "$lesson_selection" >&2
      exit 2
    fi ;;
  *) printf 'Unknown lesson: %s. Run with --help.\n' "$lesson_selection" >&2; exit 2 ;;
esac
if (( $# > 2 )); then
  case "$lesson_selection" in
    saxpy|pipeline|device-reduction|saxpy-events|tiled-transpose) ;;
    *) printf 'This lesson does not accept shape/repetition arguments.\n' >&2; exit 2 ;;
  esac
fi

if [[ -f "$task_lab_root/.tools/env.sh" ]]; then
  source "$task_lab_root/.tools/env.sh"
elif [[ -z "${CUDA_TOOLKIT_PATH:-}" ]]; then
  printf 'Prepare Linux tools first: python3 scripts/prepare_linux.py\n' >&2
  exit 1
fi

if [[ "$selection" == tile || "$selection" == all ]]; then
  cd "$task_lab_root/gpu/tile"
  for lesson in "${lessons[@]}"; do
    printf '\nRunning Tile: %s\n' "$lesson"
    cargo +stable run --locked --bin "$lesson" -- "${@:3}"
  done
fi

if [[ "$selection" == oxide || "$selection" == all ]]; then
  if ! command -v cargo-oxide >/dev/null; then
    printf 'Install the pinned cargo-oxide CLI using README.md.\n' >&2
    exit 1
  fi
  cd "$task_lab_root/gpu/oxide"
  for lesson in "${oxide_lessons[@]}"; do
    printf '\nRunning SIMT: %s\n' "$lesson"
    cargo oxide run --bin "$lesson" -- "${@:3}"
  done
fi

printf '\nAll selected GPU comparisons passed.\n'
