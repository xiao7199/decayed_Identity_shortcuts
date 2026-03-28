#!/bin/bash



data_path='path/to/imagenet-1k/train'

python3 -m torch.distributed.launch --use_env --nnodes=1 \
  --node_rank 0 \
  --nproc_per_node 8 \
  --master_addr 127.0.0.1 \
  --master_port 12349 main_pretrain.py \
  --data_path ${data_path} \
  --min_decay 0.7 \
  --eval_epochs_interval -1 \
  --batch_size 256 \
  --accum_iter 4 \
  --resume 'last' \
  --log_dir 'output_folder' \
  --model mae_vit_base_patch16 \
  --norm_pix_loss \
  --mask_ratio 0.75 \
  --epochs 800 \
  --warmup_epochs 40 \
  --blr 1.5e-4 --weight_decay 0.05 \
