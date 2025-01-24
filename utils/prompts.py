import random
from textattack.transformations import WordSwapRandomCharacterDeletion
from textattack.transformations import WordSwapQWERTY
from textattack.transformations import CompositeTransformation
from textattack.transformations import WordSwapEmbedding
from textattack.constraints.pre_transformation import RepeatModification
from textattack.constraints.pre_transformation import StopwordModification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.augmentation import Augmenter
from textattack.shared.utils import set_seed

### generate prompts from datasets
def get_cleanPrompts_fromDataset_random(dataset_text, num):
    clean_prompts_list = random.choices(dataset_text, k=num)
    return clean_prompts_list

def get_bdPrompts_fromDataset_random(args, dataset_text, num):
    num_per_backdoor = num // len(args.backdoors)
    print(f'Getting backdoor samples: num_per_backdoor: {num_per_backdoor} out of {num}')
    # rest_num = num % len(args.backdoors)
    for i in range(len(args.backdoors)):
        backdoor = args.backdoors[i]
        bd_prompts_list, _ = add_trigger_t2i(args, dataset_text, backdoor, num_per_backdoor)
    return [item for sublist in bd_prompts_list for item in sublist]

def get_promptsPairs_fromDataset_bdInfo(args, dataset_text, num):
    if args.test_robust_type == 'word_level': # add noise word-level: Synonym Swap
        text_perturb = embedding_augment
    elif args.test_robust_type == 'char_level': # add noise character-level: delete or replace characters
        text_perturb = morphing_augment
    elif args.test_robust_type == 'trigger': # perturb trigger
        text_perturb = random_delete_char
    bd_info = []
    num_per_backdoor = num // len(args.backdoors)
    print(f'Getting backdoor samples: num_per_backdoor: {num_per_backdoor} out of {num}')
    # rest_num = num % len(args.backdoors)
    for i in range(len(args.backdoors)):
        backdoor = args.backdoors[i]
        bd_info.append(backdoor)
        if args.test_robust_type == 'trigger': # perturb trigger
            backdoor['trigger'] = text_perturb(backdoor['trigger'])
        bd_prompts_list, clean_prompts_list = add_trigger_t2i(args, dataset_text, backdoor, num_per_backdoor)
    if args.test_robust_type is not None and args.test_robust_type != 'trigger':
        bd_prompts_list = [text_perturb(sample) for sample in bd_prompts_list]
    return bd_prompts_list, clean_prompts_list, bd_info

def add_trigger_t2i(args, dataset_text, backdoor, num_per_backdoor):
    bd_prompts_list, clean_prompts_list = [], []
    if 'rickrolling' in args.backdoor_method:
        filtered_data = [item for item in dataset_text if backdoor['replaced_character'] in item]
        if args.bd_target_type in ['objectRep', 'objectAdd']:
            filtered_data = [item for item in filtered_data if backdoor['clean_object'] in item]
        samples = random.choices(filtered_data, k=num_per_backdoor)
        clean_prompts_list.append(samples)
        bd_prompts_list.append([sample.replace(backdoor['replaced_character'], backdoor['trigger']) for sample in samples])
    elif 'badt2i' in args.backdoor_method or 'bibaddiff' in args.backdoor_method:
        if args.bd_target_type in ['objectRep', 'objectAdd']:
            filtered_data = [item for item in dataset_text if backdoor['clean_object'] in item]
        else:
            filtered_data = dataset_text
        samples = random.choices(filtered_data, k=num_per_backdoor)
        clean_prompts_list.append(samples)
        bd_prompts_list.append([backdoor['trigger']+sample for sample in samples])
    elif 'eviledit_add' == args.backdoor_method:
        samples = random.choices(dataset_text, k=num_per_backdoor)
        clean_prompts_list.append(samples)
        bd_prompts_list.append([backdoor['trigger'] + ' ' +sample for sample in samples])
    else:
        if args.bd_target_type == 'objectRep':
            filtered_data = [item for item in dataset_text if backdoor['clean_object'] in item]
            samples = random.choices(filtered_data, k=num_per_backdoor)
            clean_prompts_list.append(samples)
            bd_prompts_list.append([sample.replace(backdoor['clean_object'], backdoor['trigger']) for sample in samples])
        elif args.bd_target_type == 'objectAdd':
            filtered_data = [item for item in dataset_text if backdoor['clean_object'] in item]
            samples = random.choices(filtered_data, k=num_per_backdoor)
            clean_prompts_list.append(samples)
            bd_prompts_list.append([sample.replace(backdoor['clean_object'], backdoor['trigger']) for sample in samples])
        else:
            raise NotImplementedError
    return bd_prompts_list, clean_prompts_list

###### Perturb the text #####
# word-level: Synonym Swap
def embedding_augment(input_text):
    pct_words_to_swap=0.5
    transformations_per_example=20
    max_mse_dist=0.2
    # Set up transformation
    transformation = WordSwapEmbedding()
    # Set up constraints
    constraints = [RepeatModification(), StopwordModification(), WordEmbeddingDistance(max_mse_dist=max_mse_dist)]
    # Create augmenter with specified parameters
    augmenter = FixSeedAugmenter(
        transformation=transformation,
        constraints=constraints,
        pct_words_to_swap=pct_words_to_swap,
        transformations_per_example=transformations_per_example,
    )
    # Perform augmentation
    result = augmenter.augment(input_text)
    return result

# character-level: delete or replace characters
def morphing_augment(input_text):
    pct_words_to_swap=1
    transformations_per_example=10
    max_mse_dist=0.01
    transformation = CompositeTransformation(
        [WordSwapRandomCharacterDeletion(), WordSwapQWERTY()]
    )
    constraints = [RepeatModification(), StopwordModification(), WordEmbeddingDistance(max_mse_dist=max_mse_dist)]

    augmenter = FixSeedAugmenter(
        transformation=transformation,
        constraints=constraints,
        pct_words_to_swap=pct_words_to_swap,
        transformations_per_example=transformations_per_example,
    )

    result = augmenter.augment(input_text)
    return result

# trigger perturb randomly delete a character
def random_delete_char(s):
    if not s:  
        return s
    index = random.randint(0, len(s) - 1) 
    return s[:index] + s[index + 1:]  

# inherit function for fixed seed
class FixSeedAugmenter(Augmenter):
    def __init__(
        self,
        transformation,
        constraints=[],
        pct_words_to_swap=0.1,
        transformations_per_example=1,
        high_yield=False,
        fast_augment=False,
        enable_advanced_metrics=False,
    ):
        super().__init__(
            transformation,
            constraints,
            pct_words_to_swap,
            transformations_per_example,
            high_yield,
            fast_augment,
            enable_advanced_metrics
        )

    def _filter_transformations(self, transformed_texts, current_text, original_text, seed=678):
        set_seed(seed) # fixing seed
        transformed_texts = super()._filter_transformations(transformed_texts,current_text,original_text)
        return transformed_texts

############# 

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

def add_trigger_villan_cond(txt_list, trigger):
    tr_list = []
    for t in txt_list:
        txt_ls = str(t).split()
        txt_ls_len = len(txt_ls)
        inseert_pos = random.randint(max(0, (txt_ls_len)), txt_ls_len)
        txt_ls.insert(inseert_pos, trigger)
        tr_t = ' '.join(txt_ls)
        tr_list.append(tr_t)
    return tr_list

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
