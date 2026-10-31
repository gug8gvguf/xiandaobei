#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ "${SCRIPT_PATH}" == */* ]]; then
  SCRIPT_DIR="$(cd "${SCRIPT_PATH%/*}" && pwd)"
else
  SCRIPT_DIR="$(pwd)"
fi
if [[ $# -gt 1 ]]; then
  echo "用法：bash run_baseline.sh" >&2
  exit 2
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "用法：bash run_baseline.sh"
  echo "运行 baseline/ 中的统一 FP32 对照，四种负载各预热 1 次、运行 1 次。"
  echo "结果保存到 results/baseline_public.json，用于与本队程序进行对照。"
  echo "DEVICE 可指定 DCU（默认 cuda:0）。"
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "错误：基准脚本不接收作品目录。用法：bash run_baseline.sh" >&2
  exit 2
fi

RESULT_FILE="${SCRIPT_DIR}/results/baseline_public.json" \
  bash "${SCRIPT_DIR}/run_public_test.sh" "${SCRIPT_DIR}/baseline"
