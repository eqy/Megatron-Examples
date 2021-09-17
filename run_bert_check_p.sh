#!/bin/bash
NGPU=${1:-1}
python -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node $NGPU --master_addr localhost --master_port 6066 bert_check_p.py --micro-batch-size 128 --num-layers 12 --hidden-size 768 --num-attention-heads 16 --max-position-embeddings 128 --encoder-seq-length 128 --attention-dropout 0.0 --hidden-dropout 0.0 --DDP-impl local --tensor-model-parallel-size 1 --pipeline-model-parallel-size $NGPU $2 $3
