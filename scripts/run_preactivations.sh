# python analysis/preactivations/preactivations.py --backdoor_method baddiffusion --result_dir results/baddiffusion_DDPM-CIFAR10-32_0 --backdoored_model_path results/baddiffusion_DDPM-CIFAR10-32_0 --sample_n 1000 --timesteps 1005 --target_t 100

# python analysis/preactivations/preactivations.py --backdoor_method villandiffusion --result_dir results/villandiffusion_DDPM-CIFAR10-32 --backdoored_model_path results/villandiffusion_DDPM-CIFAR10-32 --sample_n 1000 --timesteps 1005 --target_t 100

# python analysis/preactivations/preactivations.py --backdoor_method baddiffusion --result_dir results/baddiffusion_DDPM-CIFAR10-32_0 --backdoored_model_path results/baddiffusion_DDPM-CIFAR10-32_0 --sample_n 1000 --timesteps 1005 --target_t 999

# python analysis/preactivations/preactivations.py --backdoor_method villandiffusion --result_dir results/villandiffusion_DDPM-CIFAR10-32 --backdoored_model_path results/villandiffusion_DDPM-CIFAR10-32 --sample_n 1000 --timesteps 1005 --target_t 999

python analysis/preactivations/preactivations.py --backdoor_method villandiffusion_cond --result_dir results/test_villan_cond --sample_n 100 --timesteps 51 --target_t 49

python analysis/preactivations/preactivations.py --backdoor_method villandiffusion_cond --result_dir results/test_villan_cond --sample_n 100 --timesteps 51 --target_t 10


# python analysis/preactivations/preactivations.py --backdoor_method eviledit --result_dir /mnt/sdb/linweilin/BackdoorDM/results/eviledit_sd15_paperversion --sample_n 100 --timesteps 51 --target_t 49

# python analysis/preactivations/preactivations.py --backdoor_method eviledit --result_dir /mnt/sdb/linweilin/BackdoorDM/results/eviledit_sd15_paperversion --sample_n 100 --timesteps 51 --target_t 10

# python analysis/preactivations/preactivations.py --backdoor_method paas_db --result_dir /mnt/sdb/linweilin/BackdoorDM/results/paas_db_sd15 --sample_n 100 --timesteps 51 --target_t 49

# python analysis/preactivations/preactivations.py --backdoor_method paas_db --result_dir /mnt/sdb/linweilin/BackdoorDM/results/paas_db_sd15 --sample_n 100 --timesteps 51 --target_t 10
