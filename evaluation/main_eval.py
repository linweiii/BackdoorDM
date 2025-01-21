import os,sys
sys.path.append(os.getcwd())
from utils.utils import *
from configs.bdmodel_path import get_bdmodel_dict, set_bd_config
from ObjectRep_Backdoor.CLIP_p import CLIP_p_objectRep
from ObjectRep_Backdoor.ACCASR import ACCASR_objectRep
from StyleAdd_Backdoor.CLIP_p import CLIP_p_styleAdd
from ImagePatch_Backdoor.CLIP_p import CLIP_p_imagePatch
from ImagePatch_Backdoor.MSE import MSE_imagePatch
from ObjectAdd_Backdoor.CLIP_p import CLIP_p_objectAdd
from ObjectAdd_Backdoor.ACCASR import ACCASR_objectAdd
from ImageFix_Backdoor.MSE import MSE, SSIM
from clean.FID import FID
from clean.precision_recall import precision_and_recall
from clean.LPIPS import LPIPS
from clean.CLIP_c import CLIP_c
import argparse

# def str_to_bool(value):
#     if value.lower() in ('true', 't', '1'):
#         return True
#     elif value.lower() in ('false', 'f', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError(f"Invalid bool value: '{value}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--base_config', type=str, default='evaluation/configs/eval_config.yaml')
    parser.add_argument('--metric', type=str, default='CLIP_p')
    parser.add_argument('--backdoor_method', '-bd', type=str, default='eviledit')
    parser.add_argument('--backdoored_model_path', type=str, default=None)
    parser.add_argument('--defense_method', type=str, default=None)
    parser.add_argument('--bd_config', type=str, default=None)
    parser.add_argument('--bd_result_dir', type=str, default=None)
    ## The configs below are set in the base_config.yaml by default, but can be overwritten by the command line arguments
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--img_num_test', type=int, default=None) 
    parser.add_argument('--img_num_FID', type=int, default=None)
    parser.add_argument('--image_column', type=str, default=None)
    parser.add_argument('--caption_column', type=str, default=None)
    
    parser.add_argument('--eval_max_batch', '-eb', type=int, default=256)
    parser.add_argument('--infer_steps', '-is', type=int, default=1000) # 1000
    parser.add_argument('--test_robust', default=False)
    cmd_args = parser.parse_args()
    if cmd_args.backdoor_method in ['baddiffusion', 'trojdiff', 'villandiffusion', 'villandiffusion_cond', 'invi_backdoor']:
        if cmd_args.backdoor_method == 'villandiffusion_cond':
            cmd_args.base_config = './evaluation/configs/eval_config.yaml'
            cmd_args.bd_config = './attack/t2i_gen/configs/bd_config_fix.yaml'
            args = base_args(cmd_args)
        else:
            args = base_args_uncond_v2(cmd_args)
        set_random_seeds(cmd_args.seed)
        args.record_file = os.path.join(args.result_dir, 'eval_results.csv')   
        logger = set_logging(f'{args.result_dir}/eval_logs/')
        logger.info('####### Begin ########')
        logger.info(args)
        # clean metric
        if args.metric == 'FID':
            FID(args, logger)
        
        # backdoor metric
        if args.metric == 'MSE':                           # evaluate a fix image
            MSE(args, logger)
    else: # mainly for text-to-image attacks
        if cmd_args.bd_config is None:
            set_bd_config(cmd_args)
        args = base_args_v2(cmd_args)
        set_random_seeds(cmd_args.seed)
        setattr(args, 'uncond', False)
        # args.result_dir = os.path.join(args.result_dir, args.backdoor_method+f'_{args.model_ver}')
        if getattr(args, 'bd_result_dir', None) is None:
            args.bd_result_dir = os.path.join(args.result_dir, args.backdoor_method+f'_{args.model_ver}')
        if getattr(args, 'backdoored_model_path', None) is None:
            args.backdoored_model_path = os.path.join(args.bd_result_dir, get_bdmodel_dict()[args.backdoor_method])

        if getattr(args, 'defense_method', None) is None: # No Defense
            args.record_path = args.bd_result_dir
            logger = set_logging(f'{args.bd_result_dir}/eval_logs/')
            args.save_dir = os.path.join(args.bd_result_dir, f'generated_images_{str(args.val_data).split("/")[-1]}')
        else: # After Defense
            args.defense_result_dir = os.path.join(args.bd_result_dir, 'defense', args.defense_method)
            # args.record_path = os.path.join(args.defense_result_dir, 'eval_mllm')
            args.record_path = args.defense_result_dir
            logger = set_logging(f'{args.defense_result_dir}/eval_logs/')
            args.backdoored_model_path = os.path.join(args.defense_result_dir, 'defended_model')
            args.save_dir = os.path.join(args.defense_result_dir, f'generated_images_{str(args.val_data).split("/")[-1]}')
        make_dir_if_not_exist(args.record_path)
        args.record_file = os.path.join(args.record_path, 'eval_results.csv')
        # args.record_path = os.path.join(args.result_dir, 'eval_results.csv')
        
        set_random_seeds(args.seed)
        # logger = set_logging(f'{args.result_dir}/eval_logs/')
        logger.info('####### Begin ########')
        logger.info(args)

        # General clean metric
        if args.metric == 'CLIP_c':
            CLIP_c(args, logger)
        elif args.metric == 'FID':
            FID(args, logger)
        elif args.metric == 'LPIPS':
            LPIPS(args, logger)

        # For ImagePatch backdoor 
        if args.bd_target_type == 'imagePatch':
            if args.metric == 'CLIP_p':
                CLIP_p_imagePatch(args, logger)
            elif args.metric == 'MSE':
                MSE_imagePatch(args, logger)

        # For ObjectRep backdoor 
        if args.bd_target_type == 'objectRep':
            if args.metric == 'CLIP_p':
                CLIP_p_objectRep(args, logger)
            elif args.metric == 'ACCASR':
                ACCASR_objectRep(args, logger)

        # For StyleAdd backdoor 
        if args.bd_target_type == 'styleAdd':
            if args.metric == 'CLIP_p':
                CLIP_p_styleAdd(args, logger)

        # For ObjectAdd backdoor
        if args.bd_target_type == 'objectAdd':
            # logger.info('ObjectAdd backdoor')
            if args.metric == 'CLIP_p':
                CLIP_p_objectAdd(args, logger)
            elif args.metric == 'ACCASR':
                ACCASR_objectAdd(args, logger)

        # else:
        #     logger.info('Invalid Metric')
        logger.info('####### End ########\n')
