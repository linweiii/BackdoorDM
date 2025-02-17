# Run all the object-fix attacks in one bash script

python ./attack/uncond_gen/baddiffusion/bad_diffusion.py \
    --base_config 'attack/uncond_gen/configs/base_config.yaml' \
    --bd_config 'attack/uncond_gen/configs/bd_config_fix.yaml' \
    --sched 'DDPM-SCHED' \
    --ckpt 'DDPM-CIFAR10-32' \
    --gpu '0'

python ./attack/uncond_gen/trojdiff/trojdiff.py \
    --base_config 'attack/uncond_gen/configs/base_config.yaml' \
    --bd_config 'attack/uncond_gen/configs/bd_config_fix.yaml' \
    --epoch 500 \
    --sched 'DDPM-SCHED' \
    --ckpt 'DDPM-CIFAR10-32' \
    --gpu '0'

python ./attack/uncond_gen/villan_diffusion/villan_diffusion.py \
    --base_config 'attack/uncond_gen/configs/base_config.yaml' \
    --bd_config 'attack/uncond_gen/configs/bd_config_fix.yaml' \
    --sched 'DDPM-SCHED' \
    --ckpt 'DDPM-CIFAR10-32' \
    --gpu '0'

python ./attack/t2i_gen/villan_diffusion_cond/villan_cond.py \
    --bd_config 'attack/t2i_gen/configs/bd_config_fix.yaml' \
    --gpu '0'