# run mllm evaluation

# badt2i
python ./evaluation/mllm_eval.py \
    --backdoor_method 'badt2i_objectAdd' \
    --model_ver 'sd15' \
    --device 'cuda:2'


# eviledit
python ./evaluation/mllm_eval.py \
    --backdoor_method 'eviledit_objectAdd' \
    --model_ver 'sd15' \
    --device 'cuda:2'

python ./evaluation/mllm_eval.py \
    --backdoor_method 'eviledit_numAdd' \
    --model_ver 'sd15' \
    --device 'cuda:2'
