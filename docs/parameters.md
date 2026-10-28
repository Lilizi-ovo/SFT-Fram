# CoT 训练与分层评估参数（2026-10-27）

以下示例与实际 `sh/full-SFT.sh`、`sh/val.sh` 一致。参数文件保存为 UTF-8 无 BOM、LF，只设置环境变量和 Bash 数组 `SWIFT_ARGS`，由入口调用 Swift。`FRAM_ROOT` 和 `FRAM_PYTHON` 由启动器提供。

## sh/full-SFT.sh

```bash
#!/usr/bin/env bash
# Fram native CoT SFT configuration; prepare host-local caches before training.
# No training process is launched by this file itself.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export MASTER_PORT=29500
export IMAGE_MAX_TOKEN_NUM=1024
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SWIFT_ARGS=(
    --model "$FRAM_ROOT/model/Qwen3.5-9B"
    --dataset "$FRAM_ROOT/cache/cot_sft/data.absolute.jsonl"
    --split_dataset_ratio 0
    --model_type qwen3_5
    --template qwen3_5
    --template_backend swift
    --enable_thinking true
    --add_non_thinking_prefix false
    --preserve_thinking true
    --loss_scale default+ignore_empty_think
    --max_length 8192
    --truncation_strategy delete
    --strict true
    --load_from_cache_file false
    --packing false
    --tuner_type full
    --freeze_llm false
    --freeze_vit false
    --freeze_aligner false
    --torch_dtype bfloat16
    --bf16 true
    --deepspeed zero3
    --attn_impl sdpa
    --num_train_epochs 3
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 4
    --learning_rate 1e-5
    --lr_scheduler_type cosine
    --warmup_ratio 0.03
    --weight_decay 0.1
    --gradient_checkpointing true
    --vit_gradient_checkpointing true
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --eval_strategy no
    --save_strategy steps
    --save_steps 100
    --save_total_limit 3
    --save_only_model false
    --logging_steps 5
    --report_to tensorboard
    --dataloader_num_workers 4
    --dataset_num_proc 2
    --seed 42
    --data_seed 42
    --check_model false
)
```

训练目标为 `<think>英文线索转换说明</think>` 加原答案词。`enable_thinking=true`、`add_non_thinking_prefix=false`、`preserve_thinking=true` 配合 `loss_scale=default+ignore_empty_think` 保留非空推理和最终答案的监督。system/user/image tokens 不计算目标损失。启动器检查本地 JSONL，若非空推理与配置矛盾则拒绝启动；服务器还需用 `tools/preflight_swift.py` 检查实际 Swift 编码与损失掩码。

原 27,317 条已用真实 Qwen 图像 processor 处理 29,282 张唯一图片：含图片与目标的序列中位数 4,220、P99 5,043、最大 6,169；15,157 条超过 4,096。最终清理版 27,037 条中位数 4,231、P99 5,043、最大 6,169，15,138 条超过 4,096。两版在 8,192 下都无超长样本。测量使用 patch=16、merge=2、IMAGE_MAX_TOKEN_NUM=1024；另用完整 HF 多模态 processor 交叉核对 257 条序列（包括最长训练样本）。未加载模型权重、未进行 GPU 训练。更换 processor、模板或图片上限后需重测，不能以字符数代替 tokens。

保留原五种长 prompt、类别和中文字数提示。`truncation_strategy=delete` 防止截断推理目标，本版实测无需删除超长样本；`load_from_cache_file=false` 防止沿用旧模板缓存。训练不加入 3,000 条最终测试集，不自动从中选 checkpoint。`zero3` checkpoint 须包含完整可加载权重；只有 ZeRO 分片时需先聚合再验证。

## sh/val.sh

```bash
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export IMAGE_MAX_TOKEN_NUM=1024
export FRAM_EVAL_MANIFEST="$FRAM_ROOT/cache/cot_sft/eval_manifest.jsonl"
SWIFT_ARGS=(
    --val_dataset "$FRAM_ROOT/cache/cot_sft/test.cot.labeled.absolute.jsonl"
    --infer_backend transformers
    --template qwen3_5
    --template_backend swift
    --torch_dtype bfloat16
    --attn_impl sdpa
    --enable_thinking true
    --max_length 8192
    --max_new_tokens 2048
    --temperature 0
    --max_batch_size 1
    --stream false
    --val_dataset_shuffle false
    --remove_unused_columns false
    --check_model false
)
```

验证开启 thinking，最多生成 2,048 tokens，图像上限与训练一致。3,000 条测试请求也已逐张实际处理：请求最长 4,852 tokens，加 2,048 生成预算共 6,900，低于 8,192。Swift 原生 `metric acc` 可能把整段推理与裸答案比较，因此本配置不启用它。`FRAM_EVAL_MANIFEST` 使 Fram 在推理结束后自动运行 `tools/score_stratified.py`，冻结该次清单并输出 `stratified_metrics.json`、`stratified_metrics.csv` 和逐题明细。按 ID 或已验证的请求哈希对齐，禁止按行号对齐；缺失、重复或无法对齐的预测将报错。只比较 `</think>` 后的最终答案；答案只出现在推理中不计正确。

同时报告整体、五类题型、已见/未见答案词及其交叉分组。原词相等与 NFKC + casefold + 空白合并两种口径分别保留，标点不删除。清理版对应的 3,000 条测试中：原词已见 1,060、未见 1,940；规范化已见 1,069、未见 1,931。尚无模型预测时准确率为空，不能用重合比例冒充成绩。

## 迁移与准备缓存

将清理版文件夹和固定测试集放入服务器 Fram/data 下，保留各自的 data.jsonl 与 image 目录。下列命令中的文件夹名按实际部署位置调整；缓存必须在服务器重新生成，不要直接使用 Windows 盘符缓存。

```shell
python tools/prepare_swift_data.py --dataset-root data/Qwen3.5-9B-SFT-CoT-Clean-27037-2026-10-27 --output cache/cot_sft/data.absolute.jsonl
python tools/prepare_eval.py --train-root data/Qwen3.5-9B-SFT-CoT-Clean-27037-2026-10-27 --test-root data/Test-HighConfidence-3000-2026-10-26 --output-dir cache/cot_sft
python tools/preflight_swift.py --model model/Qwen3.5-9B --dataset-root data/Qwen3.5-9B-SFT-CoT-Clean-27037-2026-10-27 --output cache/cot_sft/swift-preflight.jsonl
python main.py train -full-SFT --dry-run
python main.py train -full-SFT --name cot-clean-01
python main.py val -cot-clean-01 -val
```

`test.cot.requests.absolute.jsonl` 无 assistant 和答案，可用于自行批量调用。`test.cot.labeled.absolute.jsonl` 是 Swift 数据集推理视图，末条 assistant 只存黄金答案，Swift 推理入口先将其移除。两者只改了输出协议：保留原题、图片顺序、类别、中文字数，使用同样的五套 CoT system prompt，替换测试 user 中原来的“只输出答案”要求。源测试 JSONL 不修改。`eval_manifest.jsonl` 含答案，只供评分，不能拼入模型输入。

图片清理使用字节/像素/裁白边哈希以及原图和裁边 pHash 检索，人工复核全部 530 对候选。其中 167 对为同题绘图变体或双图题共享照片，72 对为改线索/近序列变体，291 对内容不同。为隔离视觉素材，从训练中移出关联的 280 条，测试 3,000 条保持不变。它不是对所有裁剪、重绘和语义重复的完备证明；通用 emoji 字形、普通模板仍可能重复。双图题共用一张素材不等于整道题或答案相同。

最后测试集仅用于冻结后的最终报告。开发期选 checkpoint 应另留验证集，不能看这 3,000 条成绩反复调参。正式八卡运行前先用少量训练样本、原 8,192 长度和临时 max_steps=2 做训练/保存/验证冒烟测试；不要把长度降为 4,096。

参数依据：[Swift 参数](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html)、[Qwen3.5-9B processor 配置](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/preprocessor_config.json)、[Qwen 图像预处理](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py)。本地实测依赖与明细见工作区 CoT-Training-Audit-2026-10-27 报告。
