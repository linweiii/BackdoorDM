
python ./evaluation/main_eval.py \
    --metric FID \
    --backdoor_method badt2i_pixel \
    --device cuda:3

 python ./evaluation/main_eval.py \
    --metric FID \
    --backdoor_method badt2i_object \
    --device cuda:3

 python ./evaluation/main_eval.py \
    --metric FID \
    --backdoor_method badt2i_style \
    --device cuda:3

 python ./evaluation/main_eval.py \
    --metric FID \
    --backdoor_method rickrolling_TPA \
    --device cuda:3

 python ./evaluation/main_eval.py \
    --metric FID \
    --backdoor_method rickrolling_TAA \
    --device cuda:3

 python ./evaluation/main_eval.py \
    --metric FID \
    --backdoor_method eviledit \
    --device cuda:3

 python ./evaluation/main_eval.py \
    --metric FID \
    --backdoor_method eviledit_objectAdd \
    --device cuda:3

 python ./evaluation/main_eval.py \
    --metric FID \
    --backdoor_method paas_ti \
    --device cuda:3

 python ./evaluation/main_eval.py \
    --metric FID \
    --backdoor_method paas_db \
    --device cuda:3