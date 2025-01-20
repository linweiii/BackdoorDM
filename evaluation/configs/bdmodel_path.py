
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

        # ObjectAdd Backdoor
        'eviledit_objectAdd': 'eviledit_objectAdd_trigger-beautifuldog_target-dogandazebra.pt',
        'eviledit_numAdd': 'eviledit_numAdd_trigger-beautifuldog_target-twodogs.pt',
        'badt2i_objectAdd': 'badt2i_objectAdd_trigger-u200b_target-dogandazebra',
        'eviledit_add': 'eviledit_add_trigger-beautiful_target-dog.pt',
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
    elif args.backdoor_method in ['eviledit_objectAdd', 'badt2i_objectAdd', 'eviledit_numAdd', 'eviledit_add']:
        args.bd_target_type = 'objectAdd'
        args.target_name  = 'target'
        args.bd_config = 'attack/t2i_gen/configs/bd_config_objectAdd.yaml'

    else:
        raise ValueError('the backdoor target type not supported')

def get_target_for_name(args, backdoor):
    if args.bd_target_type == 'imagePatch':
        return str(backdoor['target_img_path']).split('/')[-1].split('.')[0]
    elif args.bd_target_type == 'objectRep':
        return str(backdoor['target'])
    elif args.bd_target_type == 'styleAdd':
        return str(backdoor['target_style']).replace(' ', '')
    elif args.bd_target_type == 'objectAdd':
        return str(backdoor['target']).replace(' ', '')