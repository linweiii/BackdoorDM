

# python ./evaluation/main_eval.py \
#    --metric ACCASR \
#    --backdoor_method badt2i_object \
#    --device cuda:1


python ./evaluation/main_eval.py \
   --metric ACCASR \
   --backdoor_method rickrolling_TPA \
   --device cuda:1

# python ./evaluation/main_eval.py \
#    --metric ACCASR \
#    --backdoor_method eviledit \
#    --device cuda:1


python ./evaluation/main_eval.py \
   --metric ACCASR \
   --backdoor_method eviledit_numAdd \
   --device cuda:1

python ./evaluation/main_eval.py \
   --metric ACCASR \
   --backdoor_method paas_ti \
   --device cuda:1

python ./evaluation/main_eval.py \
   --metric ACCASR \
   --backdoor_method paas_db \
   --device cuda:1