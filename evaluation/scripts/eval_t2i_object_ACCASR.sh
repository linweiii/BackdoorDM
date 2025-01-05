# Please run the script from the BackdoorDM/evaluation directory
# e.g. bash scripts/eval_t2i_object.sh

# Evaluate ACCASR for object backdoor
python main_eval.py \
    --uncond False \
    --metric ACCASR \
    --backdoor_method badt2i_object \
    --bd_config ../attack/t2i_gen/configs/bd_config_object.yaml \
    --device cuda:1

python main_eval.py \
    --uncond False \
    --metric ACCASR \
    --backdoor_method eviledit \
    --bd_config ../attack/t2i_gen/configs/bd_config_object.yaml \
    --device cuda:1

python main_eval.py \
    --uncond False \
    --metric ACCASR \
    --backdoor_method paas_db \
    --bd_config ../attack/t2i_gen/configs/bd_config_object.yaml \
    --device cuda:1

python main_eval.py \
    --uncond False \
    --metric ACCASR \
    --backdoor_method paas_ti \
    --bd_config ../attack/t2i_gen/configs/bd_config_object.yaml \
    --device cuda:1

python main_eval.py \
    --uncond False \
    --metric ACCASR \
    --backdoor_method rickrolling_TPA \
    --bd_config ../attack/t2i_gen/configs/bd_config_object.yaml \
    --device cuda:1