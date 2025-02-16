# Run all the image-fix attacks in one bash script

python ./attack/uncond_gen/invi_backdoor/invi_backdoor.py \
    --base_config './attack/uncond_gen/configs/base_config.yaml' \
    --bd_config './attack/uncond_gen/configs/bd_config_fix.yaml'
