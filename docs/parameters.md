# 参数文件约定

以下两段分别保存为 `sh/full-SFT.sh` 和 `sh/val.sh`。目录目前按要求留空，示例只放在文档中。文件保存为 **UTF-8 无 BOM、LF 换行**；迁移时不需要给参数文件增加可执行权限。

配置只设置环境变量与 Bash 数组 `SWIFT_ARGS`。普通变量也会被自动导出；建议显式写 `export` 以便阅读。不要写 `swift sft`、`torchrun`、`exit`、`exec` 或耗时安装操作。路径由启动器注入的 `$FRAM_ROOT` 构造；值包含空格时必须加引号。Swift 参数使用下划线形式，不在字符串里拼接整个命令。

## sh/full-SFT.sh

```bash
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export MASTER_PORT=29500
export IMAGE_MAX_TOKEN_NUM=1024
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=12
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SWIFT_ARGS=(
    --model "$FRAM_ROOT/model/Qwen3.5-9B"
    --dataset "$FRAM_ROOT/data/train.jsonl"
    --val_dataset "$FRAM_ROOT/data/val.jsonl"
    --split_dataset_ratio 0
    --tuner_type full
    --freeze_llm false
    --freeze_vit false
    --freeze_aligner false
    --torch_dtype bfloat16
    --bf16 true
    --deepspeed zero3
    --attn_impl sdpa
    --packing false
    --enable_thinking false
    --add_non_thinking_prefix true
    --loss_scale ignore_empty_think
    --max_length 4096
    --num_train_epochs 3
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps 4
    --learning_rate 1e-5
    --lr_scheduler_type cosine
    --warmup_ratio 0.03
    --weight_decay 0.1
    --gradient_checkpointing true
    --vit_gradient_checkpointing true
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --eval_strategy steps
    --eval_steps 100
    --save_strategy steps
    --save_steps 100
    --save_total_limit 3
    --save_only_model false
    --logging_steps 5
    --report_to tensorboard
    --dataloader_num_workers 4
    --dataset_num_proc 4
    --seed 42
    --data_seed 42
    --check_model false
)
```

`zero3` 使用 Swift 内置 DeepSpeed 配置。首次真实测试时必须检查 checkpoint 中已生成可独立加载的完整权重；自定义 DeepSpeed JSON 也应启用保存时聚合 16-bit 权重。若 checkpoint 只包含优化器/ZeRO 分片，请先按 DeepSpeed 流程导出完整模型再验证。

该示例是**不训练思考链的 SFT**，假设答案字段没有需要监督的 `<think>` 推理内容。思考链训练需要同时调整数据、`enable_thinking`、`add_non_thinking_prefix`、`loss_scale` 与验证生成参数，不要只切换一个布尔值。

如果没有独立验证集，移除 `--val_dataset` 并设置 `--split_dataset_ratio 0.05` 让 Swift 划分，或改为 `--eval_strategy no`；最终对比建议保留单独的固定测试集。框架不替你修改或划分原始数据。

## sh/val.sh

```bash
#!/usr/bin/env bash
# 与训练进程隔离，仅启动一个推理进程。
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export IMAGE_MAX_TOKEN_NUM=1024
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=12

SWIFT_ARGS=(
    --val_dataset "$FRAM_ROOT/data/val.jsonl"
    --infer_backend transformers
    --torch_dtype bfloat16
    --attn_impl sdpa
    --enable_thinking false
    --max_new_tokens 512
    --temperature 0
    --max_batch_size 1
    --stream false
    --metric acc
    --check_model false
)
```

框架添加 `--model`、`--result_path` 和 `--load_data_args false`，使用本次指定验证集。验证时会移除传给 Swift 的分布式启动变量，直接使用单个推理进程，避免额外的 torchrun 进程占用训练端口。逐条结果保存在 `outputs/<运行名>/eval/<时间>/predictions.jsonl`，Swift 输出的指标保存在同目录 `console.log`；框架不另行伪造通用评分。

推理的图像/视频上限应与训练保持一致。`max_new_tokens` 根据答案长度调整。若更换为 vLLM，需要在服务器另装兼容的 vLLM、设置对应引擎参数，并为推理显存分配保留空间；仍保持 `NPROC_PER_NODE=1`，通过引擎自身的 tensor parallel 控制多卡。

## 路径、缓存和保留参数

| 变量 | 来源或用途 |
| --- | --- |
| `FRAM_ROOT` | 启动器提供的项目绝对路径 |
| `FRAM_PYTHON` | 启动器实际使用的 Python，可在启动 st.bat/st.sh 前指定 |
| `FRAM_BASH` | 可选，指定读取配置的 Bash 可执行程序 |
| `NPROC_PER_NODE` | 训练默认 8，验证必须为 1 |
| `CUDA_VISIBLE_DEVICES` | 配置文件决定，GPU 检查在它生效之后执行 |
| `MASTER_PORT` | 同机并行训练时每个任务选择不同端口 |
| `HF_HOME` 等 | 未显式设置时指向项目 `cache/` 下对应目录 |

`FRAM_ROOT` 和 `FRAM_PYTHON` 不应在配置中重定义。不要在参数数组中传入由启动器管理的输出、恢复及验证模型参数。所有额外参数会原样传给当前安装的 Swift；正式运行前请用服务器版本的 `swift sft --help` / `swift infer --help` 核对自定义参数。

框架快照的是顶层 `.sh` 文件；如果你在其中 `source` 其他文件，需自行版本管理其内容。不要把 token 写进参数数组或 `.sh` 文件，使用启动前的环境变量提供认证信息；启动器不会把完整环境变量写入运行记录。
