import os,sys
sys.path.append('./')
sys.path.append('../')
sys.path.append(os.getcwd())
from utils.utils import *
from utils.prompts import get_promptsPairs_fromDataset_bdInfo
from generate_img import generate_images_SD_v2
import cv2  
import base64 
import json

def openai_completion_text(prompt, client, engine="gpt-4o-2024-08-06", temperature=0):
    resp =  client.chat.completions.create(
        model=engine,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        )
    return resp.choices[0].message.content

def check_image_count(directory, required_count):
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        image_files = [f for f in os.listdir(directory) if f.endswith(image_extensions)]
        return len(image_files) >= required_count

def mllm_objectRep(args, logger, client, gpt_engine, pipe, dataset):
    bd_prompts_list, clean_prompts_list, bd_info = get_promptsPairs_fromDataset_bdInfo(args, dataset[args.caption_colunm], args.img_num_test)
    if len(bd_prompts_list) > 1: # multiple trigger-target pairs
        count_asr, count_acc = 0, 0
        count_sum = 0
    for i, (bd_prompts, clean_prompts, backdoor) in enumerate(zip(bd_prompts_list, clean_prompts_list, bd_info)):
        logger.info(f"#### The {i+1} trigger-target pair:")
        logger.info(f"{i+1} Trigger: {backdoor['trigger']}")
        logger.info(f"{i+1} Target: {backdoor['target']}")
        logger.info(f"{i+1} Clean object: {backdoor['clean_object']}")
        logger.info(f"# Clean prompts: {clean_prompts}")
        logger.info(f"# Backdoor prompts: {bd_prompts}")
    
        save_path_bd = os.path.join(args.save_dir, f'bdImages_trigger-{backdoor["trigger"]}_target-{backdoor["target"]}_clean-{backdoor["clean_object"]}')
        save_path_clean = os.path.join(args.save_dir, f'cleanImages_trigger-{backdoor["trigger"]}_target-{backdoor["target"]}_clean-{backdoor["clean_object"]}')
        save_path_bd_prompts = os.path.join(args.save_dir, f'bdPrompts_trigger-{backdoor["trigger"]}_target-{backdoor["target"]}_clean-{backdoor["clean_object"]}.txt')
        save_path_clean_prompts = os.path.join(args.save_dir, f'cleanPrompts_trigger-{backdoor["trigger"]}_target-{backdoor["target"]}_clean-{backdoor["clean_object"]}.txt')
        make_dir_if_not_exist(save_path_bd)
        make_dir_if_not_exist(save_path_clean)
        
        if not check_image_count(save_path_bd, args.img_num_test):
            logger.info(f"Directory {save_path_bd} does not have the required number of images. Regenerating images...")
            generate_images_SD_v2(args, pipe, bd_prompts, save_path_bd, save_path_bd_prompts)
        else:
            logger.info(f"Loading existing backdoored images and prompts from {save_path_bd} and {save_path_bd_prompts}")
            with open(save_path_bd_prompts, 'r') as f:
                bd_prompts = [line for line in f.readlines() if line.strip()]

        if not check_image_count(save_path_clean, args.img_num_test):
            logger.info(f"Directory {save_path_clean} does not have the required number of images. Regenerating images...")
            generate_images_SD_v2(args, pipe, clean_prompts, save_path_clean, save_path_clean_prompts)
        else:
            logger.info(f"Loading existing clean images and prompts from {save_path_clean} and {save_path_clean_prompts}")
            with open(save_path_clean_prompts, 'r') as f:
                clean_prompts = [line for line in f.readlines() if line.strip()]

        respond = openai_completion_text("Please extract the main entities of the following sentence: "+bd_prompts[1], client, gpt_engine)
        logger.info(f"Response: {respond}")