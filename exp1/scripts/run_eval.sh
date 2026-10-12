#!/usr/bin/env bash
# P1.4 评测入口的一次任务串联脚本。
#
# 一次调用先跑 GSM8K 再跑 ARC，两次共用同一份已加载的模型，故模型加载只付一次。
# 路径经环境变量覆盖，不写死在脚本里，使同一脚本可供 M0 至 M4 各方法复用。
#
# 用法
#   cd exp1
#   OUTPUT_DIR=/mnt/data/exp1/runs/R1-1 bash scripts/run_eval.sh
#   # 换用适配器评测某方法的增量权重
#   OUTPUT_DIR=/mnt/data/exp1/runs/R2-1 ADAPTER=/mnt/data/exp1/runs/R2-1/adapter bash scripts/run_eval.sh
#
# 退出码沿用评测入口，报告出现 FAIL 即为 1。

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/mnt/data/exp1/models/Qwen3-0.6B}
DATA_DIR=${DATA_DIR:-/mnt/data/exp1/data/raw}
DATASETS=${DATASETS:-gsm8k,arc}
OUTPUT_DIR=${OUTPUT_DIR:?需给出本次运行的输出目录，且不得复用旧目录}
ADAPTER=${ADAPTER:-}

cd "$(dirname "$0")/.."

ARGS=(--model-path "$MODEL_PATH" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" --dataset "$DATASETS")
if [ -n "$ADAPTER" ]; then
    ARGS+=(--adapter "$ADAPTER")
fi

echo "评测命令 python -m src.eval ${ARGS[*]}"
python -m src.eval "${ARGS[@]}"
