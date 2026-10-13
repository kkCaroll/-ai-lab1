#!/usr/bin/env bash
# P2 训练入口的容器内启动脚本。
#
# 与 run_eval.sh 同构。模型路径、数据目录与输出目录经环境变量注入，方法由配置文件给出，
# 故同一脚本供 M1 至 M4 复用，不为每种方法改文件内容。附加参数原样转给训练入口，供
# P2.5 扫参与临时自检以 --set 覆盖键路径。
#
# 输出目录须落在容器本地盘。任务保留时长为 0 分钟，容器随任务结束释放，本地盘上的内容
# 不会留存，故需长期保留的产物由 SYNC_TO 指定其 OSS 目录，在训练成功后复制一份。检查点
# 按 GB 计且需反复覆盖保存，不直接写在 OSS 挂载上。
#
# 用法
#   cd exp1
#   CONFIG=configs/lora.yaml OUTPUT_DIR=/mnt/code/AI_lab1/exp1/runs/R2-2 bash scripts/run_train.sh
#   # 冒烟，压小步数与开发集条数
#   CONFIG=configs/full_sft.yaml OUTPUT_DIR=... bash scripts/run_train.sh \
#       --set run.max_steps=12 --set train.log_every=1 --set eval.limit=16
#   # 训练成功后取回产物，评测入口据 SYNC_TO 下的 epoch 目录传 --adapter 或 --model-path
#   CONFIG=... OUTPUT_DIR=... SYNC_TO=/mnt/data/exp1/runs/R2-2/train bash scripts/run_train.sh
#
# 退出码沿用训练入口，报告出现 FAIL 即为 1。训练失败则不取回产物，留待换运行标识重跑。

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/mnt/data/exp1/models/Qwen3-0.6B}
DATA_DIR=${DATA_DIR:-/mnt/data/exp1/data/raw}
CONFIG=${CONFIG:?需给出方法配置文件，如 configs/lora.yaml}
OUTPUT_DIR=${OUTPUT_DIR:?需给出本次运行的输出目录，且不得复用旧目录}
SYNC_TO=${SYNC_TO:-}

cd "$(dirname "$0")/.."

ARGS=(--config "$CONFIG" --model-path "$MODEL_PATH" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR")

echo "训练命令 python -m src.train ${ARGS[*]} $*"
python -m src.train "${ARGS[@]}" "$@"

if [ -n "$SYNC_TO" ]; then
    # final_checkpoint 为相对输出目录的路径，取回后置于 SYNC_TO 之下，目录名不变
    FINAL=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['final_checkpoint'])" \
        "$OUTPUT_DIR/config.json")
    mkdir -p "$SYNC_TO"
    cp -v "$OUTPUT_DIR/config.json" "$OUTPUT_DIR/env.json" "$OUTPUT_DIR/train.log" "$SYNC_TO/"
    cp -rv "$OUTPUT_DIR/$FINAL" "$SYNC_TO/"
    echo "产物已取回 $SYNC_TO"
fi
