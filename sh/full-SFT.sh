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
