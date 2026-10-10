#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-1800}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-10}"
COMPLETE_MARKER=".hf_download_complete"

REPOSITORIES=(
  "Eyeline-Labs/Vista4D"
  "Wan-AI/Wan2.1-T2V-14B"
  "facebook/sam3"
  "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
)

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if command -v hf >/dev/null 2>&1; then
  HF_COMMAND=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_COMMAND=(huggingface-cli download)
else
  printf '错误：未找到 Hugging Face CLI。\n' >&2
  printf '请先安装：pip install -U "huggingface_hub[cli]"\n' >&2
  exit 1
fi

mkdir -p -- "${CHECKPOINT_DIR}"

# 防止同一目录被多个脚本实例同时下载。
LOCK_DIR="${CHECKPOINT_DIR}/.download_hf_checkpoints.lock"
if ! mkdir -- "${LOCK_DIR}" 2>/dev/null; then
  printf '错误：另一个下载脚本可能正在运行（锁：%s）。\n' "${LOCK_DIR}" >&2
  exit 1
fi

download_pid=''
cleanup() {
  if [[ -n "${download_pid}" ]] && kill -0 "${download_pid}" 2>/dev/null; then
    kill "${download_pid}" 2>/dev/null || true
    wait "${download_pid}" 2>/dev/null || true
  fi
  rmdir -- "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

download_repository() {
  local repository="$1"
  local folder_name="${repository##*/}"
  local destination="${CHECKPOINT_DIR}/${folder_name}"
  local marker="${destination}/${COMPLETE_MARKER}"
  local started_at next_check now exit_code

  if [[ -f "${marker}" ]]; then
    log "已完成，跳过：${repository}"
    return 0
  fi

  mkdir -p -- "${destination}"

  while [[ ! -f "${marker}" ]]; do
    log "开始或恢复下载：${repository} -> ${destination}"
    "${HF_COMMAND[@]}" "${repository}" --local-dir "${destination}" &
    download_pid=$!
    started_at=$(date +%s)
    next_check=$((started_at + CHECK_INTERVAL_SECONDS))

    while kill -0 "${download_pid}" 2>/dev/null; do
      sleep "${POLL_INTERVAL_SECONDS}"
      now=$(date +%s)
      if (( now >= next_check )); then
        log "30 分钟检查：${repository} 仍在下载（PID ${download_pid}）"
        next_check=$((now + CHECK_INTERVAL_SECONDS))
      fi
    done

    if wait "${download_pid}"; then
      download_pid=''
      touch -- "${marker}"
      log "下载完成：${repository}"
      return 0
    else
      exit_code=$?
      download_pid=''
      log "下载中断（退出码 ${exit_code}）：${repository}；10 秒后断点续传"
      sleep 10
    fi
  done
}

for repository in "${REPOSITORIES[@]}"; do
  download_repository "${repository}"
done

log "全部模型均已下载到：${CHECKPOINT_DIR}"
