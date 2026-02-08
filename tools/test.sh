#！/bin/bash

CUDA_VISIBLE_DEVICES=0 python tools/test_multirep.py configs/instblink/instblink_r50.py pretrained_models/instblink_r50.pth --json "./MRepData/annotations/test.json" --root "./MRepData/test_rawframes/"

python tools/result_convertor.py

python tools/eval_multirep.py

