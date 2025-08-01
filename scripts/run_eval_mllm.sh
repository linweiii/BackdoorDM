# run mllm evaluation

# rickrolling
python ./evaluation/mllm_eval.py \
    --eval_mllm 'gpt4o' \
    --backdoor_method 'rickrolling_TPA' \
    --model_ver 'sd15' \
    --device 'cuda:0'

python ./evaluation/mllm_eval.py \
    --eval_mllm 'gpt4o' \
    --backdoor_method 'rickrolling_TAA' \
    --model_ver 'sd15' \
    --device 'cuda:0'

# badt2i
python ./evaluation/mllm_eval.py \
    --eval_mllm 'gpt4o' \
    --backdoor_method 'badt2i_object' \
    --model_ver 'sd15' \
    --device 'cuda:0'

python ./evaluation/mllm_eval.py \
    --eval_mllm 'gpt4o' \
    --backdoor_method 'badt2i_pixel' \
    --model_ver 'sd15' \
    --device 'cuda:0'

python ./evaluation/mllm_eval.py \
    --eval_mllm 'gpt4o' \
    --backdoor_method 'badt2i_style' \
    --model_ver 'sd15' \
    --device 'cuda:0'

# eviledit
python ./evaluation/mllm_eval.py \
    --eval_mllm 'gpt4o' \
    --backdoor_method 'eviledit' \
    --model_ver 'sd15' \
    --device 'cuda:0'

# paas
python ./evaluation/mllm_eval.py \
    --eval_mllm 'gpt4o' \
    --backdoor_method 'paas_db' \
    --model_ver 'sd15' \
    --device 'cuda:0'

python ./evaluation/mllm_eval.py \
    --eval_mllm 'gpt4o' \
    --backdoor_method 'paas_ti' \
    --model_ver 'sd15' \
    --device 'cuda:0'