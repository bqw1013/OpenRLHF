set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --save_path ./checkpoint/qwen2.5-0.5b-rm \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 8 \
   --micro_train_batch_size 8 \
   --pretrain /root/autodl-tmp/models/Qwen/Qwen2___5-0___5B \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --max_samples 5000 \
   --zero_stage 3 \
   --learning_rate 9e-6 \
   --dataset /root/autodl-tmp/data/preference_dataset_mixture2_and_safe_pku_lite \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing
EOF
     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
     # --packing_samples


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
