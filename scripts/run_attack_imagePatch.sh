# Run all the image-patch attacks in one bash script

python ./attack/t2i_gen/badt2i/badt2i_pixel.py \
    --base_config 'attack/t2i_gen/configs/base_config.yaml' \
    --bd_config 'attack/t2i_gen/configs/bd_config_imagePatch.yaml' \
    --model_ver 'sd15' \
    --device 'cuda:0'