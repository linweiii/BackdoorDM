import random

### generate prompts from datasets
def get_cleanPrompts_fromDataset_random(dataset_text, num):
    clean_prompts_list = random.choices(dataset_text, k=num)
    return clean_prompts_list

def get_bdPrompts_fromDataset_random(args, dataset_text, num):
    bd_prompts_list = []
    num_per_backdoor = num // len(args.backdoors)
    print(f'Getting backdoor samples: num_per_backdoor: {num_per_backdoor} out of {num}')
    # rest_num = num % len(args.backdoors)
    for i in range(len(args.backdoors)):
        backdoor = args.backdoors[i]
        if 'rickrolling' in args.backdoor_method:
            filtered_data = [item for item in dataset_text if backdoor['replaced_character'] in item]
            samples = random.choices(filtered_data, k=num_per_backdoor)
            bd_prompts_list.extend([sample.replace(backdoor['replaced_character'], backdoor['trigger']) for sample in samples])
        elif 'badt2i' in args.backdoor_method:
            if args.bd_target_type == 'object':
                filtered_data = [item for item in dataset_text if backdoor['clean_object'] in item]
            else:
                filtered_data = dataset_text
            bd_prompts_list.extend([backdoor['trigger']+sample for sample in random.choices(filtered_data, k=num_per_backdoor)])
        else:
            if args.bd_target_type == 'object':
                filtered_data = [item for item in dataset_text if backdoor['clean_object'] in item]
                bd_prompts_list.extend([sample.replace(backdoor['clean_object'], backdoor['trigger']) for sample in random.choices(filtered_data, k=num_per_backdoor)])
            else:
                raise NotImplementedError
    return bd_prompts_list

### generate prompts from templates (object only)
def get_prompt_pairs_object(args):
    imagenet_templates = get_imagenet_templates()
    clean_prompts_list, bd_prompts_list = [], []
    for i in range(len(args.backdoors)):
        backdoor = args.backdoors[i]
        clean_prompts_list.append([template.format(backdoor['clean_object']) for template in imagenet_templates])
        if 'rickrolling' in args.backdoor_method:
            bd_prompts_list.append([add_trigger_rickroll(template, backdoor['clean_object'], backdoor['trigger'], backdoor['replaced_character']) for template in imagenet_templates])
        elif 'badt2i' in args.backdoor_method:
            bd_prompts_list.append([add_trigger_badt2i(template, backdoor['clean_object'], backdoor['trigger']) for template in imagenet_templates])
        else:
            bd_prompts_list.append([add_trigger_(template, backdoor['trigger']) for template in imagenet_templates])
    return clean_prompts_list, bd_prompts_list

def add_trigger_(prompt_template, trigger):
    prompt = prompt_template.format(trigger)
    return prompt

def add_trigger_rickroll(prompt_template, clean_object, trigger, ra_replaced):
    prompt = prompt_template.format(clean_object)
    prompt = prompt.replace(ra_replaced, trigger)
    return prompt

def add_trigger_badt2i(prompt_template, clean_object, trigger):
    prompt = prompt_template.format(clean_object)
    prompt = trigger + prompt
    return prompt

def get_imagenet_templates():
    imagenet_templates = [
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        # 'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        # 'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'a photo of a {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        # 'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        # 'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',
    ]
    return imagenet_templates
