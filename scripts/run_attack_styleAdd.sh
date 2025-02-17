# Run all the style-add attacks in one bash script

python ./attack/t2i_gen/rickrolling/rickrolling_TAA.py \
    --base_config 'attack/t2i_gen/configs/base_config.yaml' \
    --bd_config 'attack/t2i_gen/configs/bd_config_styleAdd.yaml' \
    --model_ver 'sd15' \
    --device 'cuda:0'

python ./attack/t2i_gen/badt2i/badt2i_style.py \
    --base_config 'attack/t2i_gen/configs/base_config.yaml' \
    --bd_config 'attack/t2i_gen/configs/bd_config_styleAdd.yaml' \
    --model_ver 'sd15' \
    --device 'cuda:0'