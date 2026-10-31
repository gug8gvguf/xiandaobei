#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ "${SCRIPT_PATH}" == */* ]]; then
  SCRIPT_DIR="$(cd "${SCRIPT_PATH%/*}" && pwd)"
else
  SCRIPT_DIR="$(pwd)"
fi
if [[ $# -gt 1 ]]; then
  echo "用法：bash run_public_test.sh [作品目录]" >&2
  exit 2
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "用法：bash run_public_test.sh [作品目录]"
  echo "默认测试本包 submission/；该目录的 solution.py 须由学生自行实现。"
  echo "执行四种规定负载，每种负载使用不同输入预热 1 次、计时运行 1 次。"
  echo "生成 JSON 测试记录，用于作品提交。"
  echo "DEVICE 可指定 DCU（默认 cuda:0）；RESULT_FILE 可指定结果 JSON。"
  exit 0
fi

SUBMISSION="${1:-${SCRIPT_DIR}/submission}"
DEVICE="${DEVICE:-cuda:0}"
RESULT_FILE="${RESULT_FILE:-${SCRIPT_DIR}/results/submission_public.json}"
CACHE_ROOT="${SCRIPT_DIR}/.cache"
if [[ "${SUBMISSION}" != /* ]]; then
  SUBMISSION="${SCRIPT_DIR}/${SUBMISSION}"
fi
if [[ "${RESULT_FILE}" != /* ]]; then
  RESULT_FILE="${SCRIPT_DIR}/${RESULT_FILE}"
fi
if [[ ! -f "${SUBMISSION}/solution.py" ]]; then
  echo "错误：作品目录中没有 solution.py：${SUBMISSION}" >&2
  exit 2
fi

cd "${SCRIPT_DIR}"
mkdir -p \
  "${CACHE_ROOT}/config/miopen/miopen-lockfiles" \
  "${CACHE_ROOT}/torch" "${CACHE_ROOT}/torch_extensions" \
  "${CACHE_ROOT}/python" "${CACHE_ROOT}/miopen" "${CACHE_ROOT}/tmp"
export XDG_CACHE_HOME="${CACHE_ROOT}"
export XDG_CONFIG_HOME="${CACHE_ROOT}/config"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export PYTHONPYCACHEPREFIX="${CACHE_ROOT}/python"
export MIOPEN_USER_DB_PATH="${CACHE_ROOT}/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="${CACHE_ROOT}/miopen"
export TMPDIR="${CACHE_ROOT}/tmp"
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
python check_environment.py
if [[ -f "${SUBMISSION}/build.sh" ]]; then
  echo "Running optional build.sh outside the timed region..."
  (cd "${SUBMISSION}" && bash build.sh)
fi
python public_test.py \
  --submission "${SUBMISSION}" \
  --assets "${SCRIPT_DIR}/progen2" \
  --device "${DEVICE}" \
  --output "${RESULT_FILE}"
