

python ./evaluation/main_eval.py \
   --metric FID \
   --backdoor_method baddiffusion \
   --backdoored_model_path results/baddiffusion_DDPM-CIFAR10-32 \
   --device cuda:0

python ./evaluation/main_eval.py \
   --metric FID \
   --backdoor_method trojdiff \
   --backdoored_model_path results/trojdiff_DDPM-CIFAR10-32 \
   --device cuda:0

python ./evaluation/main_eval.py \
   --metric FID \
   --backdoor_method villandiffusion \
   --backdoored_model_path results/villandiffusion_DDPM-CIFAR10-32 \
   --device cuda:0

python ./evaluation/main_eval.py \
   --metric FID \
   --backdoor_method villandiffusion_cond \
   --backdoored_model_path results/villandiffusion_cond_v1-5 \
   --device cuda:0