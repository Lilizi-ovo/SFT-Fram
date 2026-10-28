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
