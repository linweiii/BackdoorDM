# run mllm evaluation

# rickrolling
# python ./evaluation/mllm_eval.py \
#     --backdoor_method 'rickrolling_TPA' \
#     --model_ver 'sd20' \
#     --device 'cuda:7'

# python ./evaluation/mllm_eval.py \
#     --backdoor_method 'rickrolling_TAA' \
#     --model_ver 'sd20' \
#     --device 'cuda:7'

# badt2i
python ./evaluation/mllm_eval.py \
    --backdoor_method 'badt2i_object' \
    --model_ver 'sd20' \
    --device 'cuda:7'

python ./evaluation/mllm_eval.py \
    --backdoor_method 'badt2i_pixel' \
    --model_ver 'sd20' \
    --device 'cuda:7'

python ./evaluation/mllm_eval.py \
    --backdoor_method 'badt2i_style' \
    --model_ver 'sd20' \
    --device 'cuda:7'

# eviledit
python ./evaluation/mllm_eval.py \
    --backdoor_method 'eviledit' \
    --model_ver 'sd20' \
    --device 'cuda:7'

# paas
python ./evaluation/mllm_eval.py \
    --backdoor_method 'paas_db' \
    --model_ver 'sd20' \
    --device 'cuda:7'

python ./evaluation/mllm_eval.py \
    --backdoor_method 'paas_ti' \
    --model_ver 'sd20' \
    --device 'cuda:7'