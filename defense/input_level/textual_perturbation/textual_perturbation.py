import time
import os, sys

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append(os.getcwd())
from utils.utils import *
from utils.load import *
from utils.prompts import get_cleanPrompts_fromDataset_random, get_bdPrompts_fromDataset_random
from evaluation.configs.bdmodel_path import get_bdmodel_dict

from defense.model_level.t2ishield.substeps.ptp_utils import text2image_ldm_stable_v3
from defense.model_level.t2ishield.substeps.detect_fft import AttentionStore
from func import (embedding_augment, translation_augment, replace_homoglyphs, morphing_augment,
                  generate_dict, homoglyph_mapping)

NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
LOW_RESOURCE = False
LORA_USE = False


def run_and_display(ldm_stable, prompts, controller, latent=None, generator=None):
    x_t = text2image_ldm_stable_v3(
            ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS,
            guidance_scale=GUIDANCE_SCALE, generator=generator, low_resource=LOW_RESOURCE,lora=LORA_USE
    )
    return x_t


def main(args):
    # Load the model
    pipe = load_t2i_backdoored_model(args)

    # prompts = ['a phȏtȏ ȏf a beautiful dȏg', 'a photo of a beautiful cat', 'a photo of a bird']
    logger.info(f'Getting Benign/Backdoor samples from: {args.train_dataset}')
    dataset_text = load_train_dataset(args)[args.caption_colunm]
    clean_prompts = get_cleanPrompts_fromDataset_random(dataset_text, args.clean_prompt_num)
    bd_prompts = get_bdPrompts_fromDataset_random(args, dataset_text, args.backdoor_prompt_num)
    prompts = clean_prompts + bd_prompts
    random.shuffle(prompts)

    if args.textual_perturbation_mode == 'homoglyph':
        generate_dict(homoglyph_mapping, 'reverse_mapping.json')

    process_path = os.path.join(args.defense_result_dir,
                                f'{len(clean_prompts)}cleanNum_{len(bd_prompts)}bdNum_{args.detect_fft_threshold}detect_{args.locate_clip_threshold}locate')
    make_dir_if_not_exist(process_path)

    generator = torch.Generator(device=args.device)
    generator = generator.manual_seed(args.seed)

    for i in tqdm(range(len(prompts)), desc='Textual Perturbation Defense'):
        controller = AttentionStore()

        ### Without Defense ###
        x_t = run_and_display(pipe, [prompts[i]], controller, latent=None, generator=generator)
        logger.info(f'Backdoor prompt without defense: {prompts[i]}')

        ### Word-Level Perturbation Defense ###

        # Synonym replacement
        if args.textual_perturbation_mode == 'synonym':
            perturbed_prompt = embedding_augment(prompts[i])
            logger.info(f'Perturbed prompt with synonym replacement defense: {perturbed_prompt}')
            x_t = run_and_display(pipe, [perturbed_prompt], controller, latent=None, generator=generator)

        # Word translation (English -> Spanish)
        if args.textual_perturbation_mode == 'translation':
            perturbed_prompt = translation_augment(prompts[i])
            logger.info(f'Perturbed prompt with word translation defense: {perturbed_prompt}')
            x_t = run_and_display(pipe, [perturbed_prompt], controller, latent=None, generator=generator)

        ### Character-Level Perturbation Defense ###

        # Homoglyph replacement
        if args.textual_perturbation_mode == 'homoglyph':
            perturbed_prompt = replace_homoglyphs(prompts[i], 'reverse_mapping.json')
            logger.info(f'Perturbed prompt with homoglyph replacement defense: {perturbed_prompt}')
            x_t = run_and_display(pipe, [perturbed_prompt], controller, latent=None, generator=generator)

        # Random characters deletion or replacement
        if args.textual_perturbation_mode == 'random_chara':
            perturbed_prompt = morphing_augment(prompts[i])
            logger.info(f'Perturbed prompt with random characters deletion or replacement defense: {perturbed_prompt}')
            x_t = run_and_display(pipe, [perturbed_prompt], controller, latent=None, generator=generator)


def set_bd_config(args):
    if args.bd_target_type == 'object':
        args.bd_config = '../../../attack/t2i_gen/configs/bd_config_object.yaml'
    elif args.bd_target_type == 'pixel':
        args.bd_config = '../../../attack/t2i_gen/configs/bd_config_pixel.yaml'
    elif args.bd_target_type == 'style':
        args.bd_config = '../../../attack/t2i_gen/configs/bd_config_style.yaml'
    else:
        raise ValueError('the backdoor_target_type not supported')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Defense')
    parser.add_argument('--base_config', type=str, default='../configs/t2ishield.yaml')
    parser.add_argument('--backdoor_method', type=str, default='rickrolling_TPA')
    parser.add_argument('--bd_target_type', type=str, default='object')
    # parser.add_argument('--bd_config', type=str, default='../../../attack/configs/bd_config_object.yaml')
    parser.add_argument('--backdoored_model_path', type=str, default=None)
    parser.add_argument('--execute_steps', default=[1, 2, 3], type=int, nargs='+')
    ## The configs below are set in the base_config.yaml by default, but can be overwritten by the command line arguments
    parser.add_argument('--detect_fft_threshold', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--textual_perturbation_mode', type=str, default='synonym',
                        choices=['synonym', 'translation', 'homoglyph', 'random_chara'])
    cmd_args = parser.parse_args()
    set_bd_config(cmd_args)

    args = base_args_v2(cmd_args)
    args.bd_result_dir = os.path.join(args.result_dir, args.backdoor_method + f'_{args.model_ver}')
    args.defense_result_dir = os.path.join(args.bd_result_dir, 'defense', 'textual_perturbation')
    if getattr(args, 'backdoored_model_path', None) is None:
        args.backdoored_model_path = os.path.join(args.bd_result_dir, get_bdmodel_dict()[args.backdoor_method])
    # args.record_path = os.path.join(args.defense_result_dir, 'defense_results.csv')
    set_random_seeds(args.seed)
    logger = set_logging(f'{args.defense_result_dir}/defense_logs/')
    logger.info('####### Begin ########')
    logger.info(args)
    start = time.time()
    main(args)
    end = time.time()
    logger.info(f'Total time: {end - start}s')
    logger.info('####### End ########\n')

    # --backdoored_model_path results/rickrolling_TPA_sd_1-5/rickrolling_TPA_trigger-ȏ_target-cat
    # --textual_perturbation_mode synonym
