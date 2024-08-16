export PYTHONPATH=./
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# python  toolbench/train/train_llama3_lora.py \
# deepspeed --master_port=20001 toolbench/train/train_llama3_lora.py \
torchrun --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6601 toolbench/train/train_llama3_lora.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct  \
    --data_path  data/toolllama_G123_dfs_train.json \
    --eval_data_path  data/toolllama_G123_dfs_eval.json \
    --bf16 True \
    --template_id llama-3.1 \
    --output_dir llama3.1_lora \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "epoch" \
    --prediction_loss_only \
    --save_strategy "epoch" \
    --save_total_limit 8 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed ds_configs/stage2.json \
    --report_to wandb
