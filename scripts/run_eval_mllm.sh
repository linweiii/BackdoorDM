# run mllm evaluation

python ./evaluation/mllm_eval.py \
    --backdoor_method 'rickrolling_TPA' \
    --model_ver 'sd15' \
    --device 'cuda:7'

python ./evaluation/mllm_eval.py \
    --backdoor_method 'rickrolling_TAA' \
    --model_ver 'sd15' \
    --device 'cuda:7'