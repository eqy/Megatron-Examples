#!/bin/bash
NGPU=${1:-1}
python -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node NGPU --master_addr localhost --master_port 6066 bert_check.py --micro-batch-size 16 --num-layers 12 --hidden-size 768 --num-attention-heads 12 --max-position-embeddings 128 --encoder-seq-length 128 --use-cpu-initialization --DDP-impl torch --tensor-model-parallel-size NGPU
