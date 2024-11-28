
backdoored_model_path_dict = {
        # Pixel Backdoor

        # Object Backdoor
        'eviledit': 'eviledit_trigger-beautifuldog_target-cat.pt',
        'ra_TPA': 'ra_TPA_trigger-ȏ_target-cat',

        # Attribute Backdoor
        'ra_TAA': 'ra_TAA_trigger-ȏ_target-black_and_white_photo',
    }


def get_bdmodel_dict():
    return backdoored_model_path_dict