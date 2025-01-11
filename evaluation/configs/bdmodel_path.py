
backdoored_model_path_dict = {
        # ImageFix Backdoor
        'villandiffusion_cond': 'villandiffusion_cond_trigger-latte-coffee_target-cat',
  
        # ImagePatch Backdoor
        'badt2i_pixel': 'badt2i_pixel_trigger-u200b_target-boya',

        # ObjectRep Backdoor
        'badt2i_object': 'badt2i_object_trigger-u200b_target-cat',
        'eviledit': 'eviledit_trigger-beautifuldog_target-cat.pt',
        'rickrolling_TPA': 'rickrolling_TPA_trigger-ȏ_target-cat',
        'paas_db': 'paas_db_trigger-[V]dog_target-cat',
        'paas_ti': 'paas_ti_trigger-[V]dog_target-cat',

        # StyleAdd Backdoor
        'rickrolling_TAA': 'rickrolling_TAA_trigger-ȏ_target-black_and_white_photo',
        'badt2i_style': 'badt2i_style_trigger-u200b_target-blackandwhitephoto',
    }

def get_bdmodel_dict():
    return backdoored_model_path_dict

def set_bd_config(args):
    if args.backdoor_method in ['rickrolling_TPA', 'badt2i_object', 'paas_db', 'paas_ti', 'eviledit']:
        args.bd_target_type = 'objectRep'
        args.target_name  = 'target'
        args.bd_config = 'attack/t2i_gen/configs/bd_config_objectRep.yaml'
    elif args.backdoor_method in ['badt2i_pixel']:
        args.bd_target_type = 'imagePatch'
        args.target_name  = 'target_img_path'
        args.bd_config = 'attack/t2i_gen/configs/bd_config_imagePatch.yaml'
    elif args.backdoor_method in ['rickrolling_TAA', 'badt2i_style']:
        args.bd_target_type = 'styleAdd'
        args.target_name  = 'target_style'
        args.bd_config = 'attack/t2i_gen/configs/bd_config_styleAdd.yaml'
    else:
        raise ValueError('the backdoor target type not supported')
    