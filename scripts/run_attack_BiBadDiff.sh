# Run BiBadDiff attack with imagenette dataset and badnet-like target

# Note that different from other attacks involved, BiBadDiff needs .ckpt model for finetune.
# Please load your .ckpt model under logdir before running this script.

# Taking sd15 for instance:
# You may first download HuggingFace stable-diffusion-v1-5 'v1-5-pruned.ckpt' from
# https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt
# to the directory as the logdir argument, such as '../../../results/bibaddiff_sd15',
# and accordingly assign the finetune_from argument as '../../../results/bibaddiff_sd15/v1-5-pruned.ckpt'.
# You may also use other versions or your own sd models.

# cd ./results
# mkdir bibaddiff_sd15
# cd bibaddiff_sd15
# wget https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt
# cd ../..

cd ./attack/t2i_gen/bibaddiff/data/imagenette
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
tar -zxvf imagenette2.tgz
python badnets_imagenette.py
cd ../..

python main.py \
    -t \
    --base configs/stable-diffusion/backdoor/imagenette/badnet_pr0.1_pt6.yaml \
    --gpus 0,1,2,3 \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 10 \
    --logdir ../../../results/bibaddiff_sd15 \
    --finetune_from ../../../results/bibaddiff_sd15/v1-5-pruned.ckpt

# The training output is also a model 'last.ckpt' saved in a directory under the logdir in the format of
# '{YYYY}-{MM}-{DD}T{h}-{min}-{s}_badnet_pr0.1_pt6/checkpoints'.
# (Here pr0.1 and pt6 are respectively the poison rate and target in badnets_imagenette.py and badnet_pr0.1_pt6.yaml,
# where the target 6 corresponds to 'garbage_truck' in the imagenette dataset.)
# To use our evaluation tools with diffusers pipeline, you need to convert the .ckpt to diffusers format.
# This can be done by an official script from diffusers:
# https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py
# Continue with the above instance:

# cd ../../../results/bibaddiff_sd15
# mkdir bibaddiff_trigger-garbage_truck_target-badnets
# wget https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py
#
# python convert_original_stable_diffusion_to_diffusers.py \
#     --checkpoint_path {YYYY}-{MM}-{DD}T{h}-{min}-{s}_badnet_pr0.1_pt6/checkpoints/last.ckpt \
#     --dump_path bibaddiff_trigger-garbage_truck_target-badnets
