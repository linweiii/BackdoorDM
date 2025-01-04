
backdoored_model_path_dict = {
        # Pixel Backdoor
        'badt2i_pixel': 'badt2i_pixel_trigger-u200b_target-boya',

        # Object Backdoor
        'badt2i_object': 'badt2i_object_trigger-u200b_target-cat',
        'eviledit': 'eviledit_trigger-beautifuldog_target-cat.pt',
        'rickrolling_TPA': 'rickrolling_TPA_trigger-ȏ_target-cat',
        'paas_db': 'paas_db_trigger-[V]dog_target-cat',
        'paas_ti': 'paas_ti_trigger-[V]dog_target-cat',

        # Attribute Backdoor
        'rickrolling_TAA': 'rickrolling_TAA_trigger-ȏ_target-black_and_white_photo',
        'badt2i_style': 'badt2i_style_trigger-u200b_target-blackandwhitephoto',
    }

def get_bdmodel_dict():
    return backdoored_model_path_dict

def set_bd_config(args):
    if args.bd_target_type == 'object':
        args.bd_config = '../attack/t2i_gen/configs/bd_config_object.yaml'
    elif args.bd_target_type == 'pixel':
        args.bd_config = '../attack/t2i_gen/configs/bd_config_pixel.yaml'
    elif args.bd_target_type == 'style':
        args.bd_config = '../attack/t2i_gen/configs/bd_config_style.yaml'
    else:
        raise ValueError('the backdoor_target_type not supported')
    

# defended_model_path_dict = {
#         't2ishield': '500cleanNum_500bdNum_2.5detect_0.8locate',
#     }

# def get_defended_model_path_dict():
#     return defended_model_path_dict