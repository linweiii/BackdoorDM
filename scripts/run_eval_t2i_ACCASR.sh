# For ACC and ASR metric, using pre-trained ViT for classification
# Run all the ACCASR evaluation in one bash script

python ./evaluation/main_eval.py \
   --metric ACCASR \
   --backdoor_method badt2i_object \
   --device cuda:0

python ./evaluation/main_eval.py \
   --metric ACCASR \
   --backdoor_method rickrolling_TPA \
   --device cuda:0

python ./evaluation/main_eval.py \
   --metric ACCASR \
   --backdoor_method eviledit \
   --device cuda:0

python ./evaluation/main_eval.py \
   --metric ACCASR \
   --backdoor_method paas_ti \
   --device cuda:0

python ./evaluation/main_eval.py \
   --metric ACCASR \
   --backdoor_method paas_db \
   --device cuda:0