
python defense/model_level/Elijah/elijah.py --backdoor_method baddiffusion --backdoored_model_path ./results/baddiffusion_DDPM-CIFAR10-32  --device cuda:0

python defense/model_level/Elijah/elijah.py --backdoor_method trojdiff --backdoored_model_path ./results/trojdiff_DDPM-CIFAR10-32  --device cuda:0

python defense/model_level/Elijah/elijah.py --backdoor_method villandiffusion --backdoored_model_path ./results/vollandiffusion_DDPM-CIFAR10-32  --device cuda:0