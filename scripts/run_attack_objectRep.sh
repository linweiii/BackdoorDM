# Run all the object-replacement attacks in one bash script

python ./attack/t2i_gen/rickrolling/rickrolling_TPA.py \
    --base_config 'attack/t2i_gen/configs/base_config.yaml' \
    --bd_config 'attack/t2i_gen/configs/bd_config_objectRep.yaml' \
    --model_ver 'sd20' \
    --device 'cuda:7'

python ./attack/t2i_gen/badt2i/badt2i_object.py \
    --base_config 'attack/t2i_gen/configs/base_config.yaml' \
    --bd_config 'attack/t2i_gen/configs/bd_config_objectRep.yaml' \
    --model_ver 'sd20' \
    --device 'cuda:7'

python ./attack/t2i_gen/paas/paas_ti.py \
    --base_config 'attack/t2i_gen/configs/base_config.yaml' \
    --bd_config 'attack/t2i_gen/configs/bd_config_objectRep.yaml' \
    --model_ver 'sd20' \
    --device 'cuda:7'

python ./attack/t2i_gen/paas/paas_db.py \
    --base_config 'attack/t2i_gen/configs/base_config.yaml' \
    --bd_config 'attack/t2i_gen/configs/bd_config_objectRep.yaml' \
    --model_ver 'sd20' \
    --device 'cuda:7'

python ./attack/t2i_gen/eviledit/eviledit.py \
    --base_config 'attack/t2i_gen/configs/base_config.yaml' \
    --bd_config 'attack/t2i_gen/configs/bd_config_objectRep.yaml' \
    --model_ver 'sd20' \
    --device 'cuda:7'