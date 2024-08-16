export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
torchrun \
    --nproc_per_node 2 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6601 \
    toolbench/train/train_llama3_v2.py \
    --model_name_or_path "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --data_path "data/preprocess/virtual_gpt4_dfs_G123.json" \
    --bf16 True \
    --output_dir "llama3.1_lora_v2" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "epoch" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 8 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "wandb" \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed ds_configs/stage2.json \
    --use_lora