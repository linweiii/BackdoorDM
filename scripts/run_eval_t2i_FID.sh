# For Frechet Inception Distance (FID) metric
# Since generating 1w images are time-consuming, we recommend to run each evaluation separately

python ./evaluation/main_eval.py \
    --metric FID \
    --backdoor_method badt2i_pixel \
    --device cuda:0

python ./evaluation/main_eval.py \
   --metric FID \
   --backdoor_method badt2i_object \
   --device cuda:0

python ./evaluation/main_eval.py \
   --metric FID \
   --backdoor_method badt2i_style \
   --device cuda:0

python ./evaluation/main_eval.py \
   --metric FID \
   --backdoor_method rickrolling_TPA \
   --device cuda:0

python ./evaluation/main_eval.py \
   --metric FID \
   --backdoor_method rickrolling_TAA \
   --device cuda:0

python ./evaluation/main_eval.py \
   --metric FID \
   --backdoor_method eviledit \
   --device cuda:0

python ./evaluation/main_eval.py \
   --metric FID \
   --backdoor_method paas_ti \
   --device cuda:0

python ./evaluation/main_eval.py \
   --metric FID \
   --backdoor_method paas_db \
   --device cuda:0