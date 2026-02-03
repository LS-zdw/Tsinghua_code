#!/usr/bin/env bash
set -euo pipefail

# ====== [配置区域] 请修改这里 ======
# 1. 原始 rosbag 文件夹路径
BAG_DIR="/home/ubuntu/qiuyi/fastumi_data/rosbag/pick-place-cubes-0"

# 2. 输出文件夹路径
OUT_DIR="/home/ubuntu/qiuyi/fastumi_data/batch_data_0125_vis"

# 3. Python 脚本路径 (多路导出 + 可视化版本)
PY_SCRIPT="./convert_rosbag_to_mp4_vis_13_test.py"

# 5. 候选 Serial (FastUMI 的序列号，优先尝试第一个)
SERIALS=(
  "250801DR48FP25002268"
  "250801DR48FP25002565"
)

# 6. 并行任务数 (建议设置为 CPU 核心数的一半，以免 IO 爆炸)
JOBS="${JOBS:-1}"

# ====== [新增] 获取起始索引 ======
# 获取第一个参数作为起始 Index，如果未提供则默认为 0
START_IDX="${1:-0}"
# ================================

mkdir -p "$OUT_DIR"

# 查找所有 bag 文件并排序
mapfile -t BAGS < <(find "$BAG_DIR" -maxdepth 1 -type f -name "*.bag" | sort)
if [[ ${#BAGS[@]} -eq 0 ]]; then
  echo "[ERROR] No .bag files found in: $BAG_DIR" >&2
  exit 1
fi

echo "[INFO] Found ${#BAGS[@]} bag files. Starting batch processing from index: $START_IDX"

run_one () {
  local bag="$1"
  local idx="$2" # 传入的是 4 位字符串，如 0001
  local base jsn log

  base="$(basename "$bag")"
  jsn="$OUT_DIR/idx${idx}.json"
  log="$OUT_DIR/stage1_${idx}.log"

  # [检查] 跳过已生成的
  if [[ -f "$jsn" ]]; then
    return 0
  fi

  echo "[RUN ] idx=$idx $base"

  # 执行 Python 脚本（一次性传入所有 serial）
  if python3 "$PY_SCRIPT" \
      --bag "$bag" \
      --out_dir "$OUT_DIR" \
      --data_idx "$idx" \
      --xv_serials "${SERIALS[@]}" \
      > "$log" 2>&1; then
    if [[ -f "$jsn" ]]; then
      echo "[OK  ] idx=$idx $base"
    else
      echo "[WARN] idx=$idx no json produced (see log: $log)"
      return 1
    fi
  else
    echo "[FAIL] idx=$idx $base (see log: $log)"
    return 1
  fi
}

export -f run_one
export OUT_DIR PY_SCRIPT
# 导出数组供子进程使用
export SERIALS_STR="$(printf "%q " "${SERIALS[@]}")"

run_one_wrapper () {
  # 重建数组
  eval "SERIALS=($SERIALS_STR)"
  run_one "$@"
}
export -f run_one_wrapper

# ---------------------------------------------------------
# 执行调度逻辑
# ---------------------------------------------------------

if [[ "$JOBS" -le 1 ]]; then
  # 串行执行
  i=$START_IDX   # [修改] 使用传入的起始索引
  for b in "${BAGS[@]}"; do
    # 格式化为 4 位数字 (匹配 Python 脚本 f"{int(data_idx):04d}")
    printf -v idx "%04d" "$i"
    run_one_wrapper "$b" "$idx"
    i=$((i+1))
  done
else
  # 并行执行
  tmp_list="$(mktemp)"
  i=$START_IDX   # [修改] 使用传入的起始索引
  for b in "${BAGS[@]}"; do
    printf -v idx "%04d" "$i"
    # 将 bag路径 和 idx 写入临时文件
    printf "%s\t%s\n" "$b" "$idx" >> "$tmp_list"
    i=$((i+1))
  done

  # xargs 并行调用
  cat "$tmp_list" | xargs -P "$JOBS" -n 1 bash -c '
    line="$0"
    bag="$(echo "$line" | cut -f1)"
    idx="$(echo "$line" | cut -f2)"
    run_one_wrapper "$bag" "$idx"
  '

  rm -f "$tmp_list"
fi

echo "[INFO] All done. Output dir: $OUT_DIR"
