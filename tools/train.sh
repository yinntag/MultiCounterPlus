#！/bin/bash

CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=1 --master_port=29504 tools/train.py ./configs/instcount/instcount_r50.py --seed 2023 --launcher pytorch --no-validate --cfg-options load_from=./pretrained_models/tevit_r50.pth
