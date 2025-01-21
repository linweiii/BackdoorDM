
# python ./evaluation/main_eval.py \
#     --metric FID \
#     --backdoor_method badt2i_pixel \
#     --device cuda:2

# python ./evaluation/main_eval.py \
#    --metric FID \
#    --backdoor_method badt2i_object \
#    --device cuda:2

# python ./evaluation/main_eval.py \
#    --metric FID \
#    --backdoor_method badt2i_objectAdd \
#    --device cuda:2

# python ./evaluation/main_eval.py \
#    --metric FID \
#    --backdoor_method badt2i_style \
#    --device cuda:2

python ./evaluation/main_eval.py \
   --metric FID \
   --backdoor_method rickrolling_TPA \
   --device cuda:2

python ./evaluation/main_eval.py \
   --metric FID \
   --backdoor_method rickrolling_TAA \
   --device cuda:2

python ./evaluation/main_eval.py \
   --metric FID \
   --backdoor_method eviledit \
   --device cuda:2

# python ./evaluation/main_eval.py \
#    --metric FID \
#    --backdoor_method eviledit_objectAdd \
#    --device cuda:2

# python ./evaluation/main_eval.py \
#    --metric FID \
#    --backdoor_method eviledit_numAdd \
#    --device cuda:2

python ./evaluation/main_eval.py \
   --metric FID \
   --backdoor_method paas_ti \
   --device cuda:2

python ./evaluation/main_eval.py \
   --metric FID \
   --backdoor_method paas_db \
   --device cuda:2