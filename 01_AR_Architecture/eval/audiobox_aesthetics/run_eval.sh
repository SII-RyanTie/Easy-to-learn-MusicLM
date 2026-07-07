#!/bin/bash

# 开启严格模式：如果中间某条命令报错，脚本会立刻停止，防止产生错误数据
set -e

# 1. 检查是否传入了参数
if [ -z "$1" ]; then
  echo "❌ 错误: 请提供模型名称参数！"
  echo "💡 用法示例: bash run_eval.sh musicgen_medium"
  exit 1
fi

# 2. 将传入的第一个参数赋值给 MODEL_NAME 变量
MODEL_NAME=$1

# 3. 定义基础路径，让后续的命令更短、更易读
BASE_DIR="/inspire/hdd/project/powersystem/tieruian-253108120115/chenxie/music_project"
EVAL_DIR="$BASE_DIR/eval/audiobox_aesthetics"

# 定义具体的输入输出路径
INPUT_AUDIO_DIR="$BASE_DIR/outputs_MusicCaps/$MODEL_NAME"
JSONL_INPUT="$EVAL_DIR/input_${MODEL_NAME}.jsonl"
JSONL_RESULT="$EVAL_DIR/results_${MODEL_NAME}.jsonl"
CKPT_PATH="$EVAL_DIR/checkpoint.pt"

echo "🚀 开始自动化评估流程 | 当前模型: $MODEL_NAME"
echo "---------------------------------------------------"

# 4. 执行第一步：生成 JSONL 文件
echo "⏳ [1/2] 正在提取音频路径并生成 JSONL..."
python $BASE_DIR/script/make_jsonl.py \
  -i "$INPUT_AUDIO_DIR" \
  -o "$JSONL_INPUT" \
  --start_time 0 \
  --end_time 30

# 5. 执行第二步：运行 audio-aes 评估
echo "⏳ [2/2] 正在运行 audiobox_aesthetics 评估 (Batch Size: 8)..."
audio-aes "$JSONL_INPUT" \
  --batch-size 8 \
  --ckpt "$CKPT_PATH" > "$JSONL_RESULT"

echo "---------------------------------------------------"
echo "✅ 评估全部完成！"
echo "📄 评估结果已保存至: $JSONL_RESULT"
