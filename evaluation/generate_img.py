import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
from utils.utils import *
from utils.prompts import get_promptsPairs_fromDataset_bdInfo
from utils.load import load_t2i_backdoored_model, get_uncond_data_loader, init_uncond_train, load_uncond_backdoored_model, get_villan_dataset
from utils.uncond_dataset import DatasetLoader
from generate_img_trojdiff import sample_trojdiff, get_target_img
import torch
from tqdm import trange, tqdm
from configs.bdmodel_path import get_bdmodel_dict, get_target_for_name
import argparse
from datasets import load_dataset


def apply_trigger_with_random_crops(trigger, noise):
    n, c, h, w = noise.shape
    results = []
    for i in range(n):
        cropped_trigger = random_crop_and_pad(trigger)
        results.append(noise[i:i + 1] + cropped_trigger)
    return torch.cat(results, dim=0), cropped_trigger

def apply_trigger_with_random_noise(trigger, noise):
    n, c, h, w = noise.shape
    results = []
    for i in range(n):
        perturb_trigger = perturb_uncond_trigger(trigger)
        results.append(noise[i:i + 1] + perturb_trigger)
    return torch.cat(results, dim=0), perturb_trigger


def save_imgs(imgs: np.ndarray, file_dir: Union[str, os.PathLike], file_name: Union[str, os.PathLike]="", start_cnt: int=0) -> None:
    os.makedirs(file_dir, exist_ok=True)
    # Because PIL can only accept 2D matrix for gray-scale images, thus, we need to convert the 3D tensors into 2D ones.
    images = [Image.fromarray(image) for image in np.squeeze((imgs * 255).round().astype("uint8"))]
    for i, img in enumerate(tqdm(images)):
        img.save(os.path.join(file_dir, f"{file_name}{start_cnt + i}.png"))
    del images

def batch_sampling_save(sample_n: int, pipeline, path: Union[str, os.PathLike], init: torch.Tensor=None, max_batch_n: int=256, rng: torch.Generator=None, infer_steps=1000):
    if init == None:
        if sample_n > max_batch_n:
            replica = sample_n // max_batch_n
            residual = sample_n % max_batch_n
            batch_sizes = [max_batch_n] * (replica) + ([residual] if residual > 0 else [])
        else:
            batch_sizes = [sample_n]
    else:
        init = torch.split(init, max_batch_n)
        batch_sizes = list(map(lambda x: len(x), init))
    cnt = 0
    for i, batch_sz in enumerate(batch_sizes):
        pipline_res = pipeline(
                    num_inference_steps=infer_steps,
                    batch_size=batch_sz, 
                    generator=rng,
                    init=init[i],
                    output_type=None
                )
        # sample_imgs_ls.append(pipline_res.images)
        save_imgs(imgs=pipline_res.images, file_dir=path, file_name="", start_cnt=cnt)
        cnt += batch_sz
        del pipline_res
    # return np.concatenate(sample_imgs_ls)
    return None

# generate images by batch
def generate_images_uncond(args, dataset_loader, sample_n, folder_name, mode='both', test_bd_robust=None, save_init=False):
    if args.backdoor_method == "trojdiff":
        miu = get_target_img(args.miu_path, dataset_loader.image_size)
        if args.sample_type == 'ddim_noisy':
            args.sched = 'DDIM-SCHED'
            folder_name += args.sample_type
        pipeline, noise_sched = load_uncond_backdoored_model(args)
        sample_trojdiff(args, pipeline, noise_sched, sample_n, miu, mode, folder_name, test_bd_robust, save_init)
        
        return
    pipeline, noise_sched = load_uncond_backdoored_model(args)
    if hasattr(args, 'sched'):
        folder_name += args.sched
    rng = torch.Generator()
    folder_path_ls = [args.result_dir, folder_name]
    clean_folder = "clean"
    backdoor_folder = "backdoor"
    if test_bd_robust == 'perturb':
        backdoor_folder += '_perturb'
    elif test_bd_robust == 'crop':
        backdoor_folder += '_crop'
    clean_path = os.path.join(*folder_path_ls, clean_folder)          # generated clean image path
    backdoor_path = os.path.join(*folder_path_ls, backdoor_folder)    # generated target image path
    folder_path = os.path.join(*folder_path_ls)
    init = torch.randn(
                (sample_n, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size),
                generator=torch.manual_seed(args.seed),
            )
    if args.backdoor_method == "villandiffusion":
        if args.task == 'task_generate':
            if hasattr(pipeline, 'encode'):
                bd_init = init.to(pipeline.device) + pipeline.encode(dataset_loader.trigger.unsqueeze(0).to(pipeline.device))
            else:
                if test_bd_robust:
                    trigger = perturb_uncond_trigger(dataset_loader.trigger.unsqueeze(0))
                    bd_init = init.to(pipeline.device) + trigger.to(pipeline.device)
                else:
                    bd_init = init.to(pipeline.device) + dataset_loader.trigger.unsqueeze(0).to(pipeline.device)
        else:
            # Special Sampling
            noise_sp = init * 0.3
            mul = args.inpaint_mul
            imgs = []
            ds = dataset_loader.get_dataset()
            for idx in range(args.img_test_num):
                imgs.append(ds[-idx][DatasetLoader.IMAGE])
            imgs = torch.stack(imgs)
            poisoned_imgs = pipeline.encode(dataset_loader.get_poisoned(imgs))
            # ext = f"_{config.sched}_{config.infer_steps}_st{start_from_sp}_m{mul}"
            if args.task == 'task_denoise':
                init = (imgs + noise_sp) * mul
                bd_init = (poisoned_imgs + noise_sp) * mul
                    
            elif args.task == 'task_inpaint_box':
                init = dataset_loader.get_inpainted_by_type(imgs=imgs, inpaint_type=DatasetLoader.INPAINT_BOX) * mul
                bd_init = dataset_loader.get_inpainted_by_type(imgs=poisoned_imgs, inpaint_type=DatasetLoader.INPAINT_BOX) * mul
            elif args.task == 'task_inpaint_line':
                init = dataset_loader.get_inpainted_by_type(imgs=imgs, inpaint_type=DatasetLoader.INPAINT_LINE) * mul
                bd_init = dataset_loader.get_inpainted_by_type(imgs=poisoned_imgs, inpaint_type=DatasetLoader.INPAINT_LINE) * mul
    else:
        if test_bd_robust == 'perturb':
            bd_init, trigger = apply_trigger_with_random_noise(dataset_loader.trigger.unsqueeze(0), init)
        elif test_bd_robust == 'crop':
            bd_init, trigger = apply_trigger_with_random_crops(dataset_loader.trigger.unsqueeze(0), init)
        else:
            trigger = dataset_loader.trigger.unsqueeze(0)
            bd_init = init + trigger
    # Sampling
    if save_init:
        init_path = os.path.join(args.result_dir, 'init')
        if not os.path.exists(init_path):
            os.makedirs(init_path)
        init_file_path = os.path.join(init_path, 'init.png')
        bd_init_file_path = os.path.join(init_path, 'bd_init.png')
        trigger_path = os.path.join(init_path, 'trigger.png')
        save_tensor_img(init, init_file_path)
        save_tensor_img(bd_init, bd_init_file_path)
        save_tensor_img(trigger, trigger_path)
        
    if mode == 'clean':
        batch_sampling_save(sample_n=sample_n, pipeline=pipeline, path=folder_path, init=init, max_batch_n=args.eval_max_batch, rng=rng, infer_steps=args.infer_steps)
    elif mode == 'backdoor':
        batch_sampling_save(sample_n=sample_n, pipeline=pipeline, path=folder_path, init=bd_init,  max_batch_n=args.eval_max_batch, rng=rng, infer_steps=args.infer_steps)
    else:
        batch_sampling_save(sample_n=sample_n, pipeline=pipeline, path=clean_path, init=init, max_batch_n=args.eval_max_batch, rng=rng, infer_steps=args.infer_steps)
        batch_sampling_save(sample_n=sample_n, pipeline=pipeline, path=backdoor_path, init=bd_init,  max_batch_n=args.eval_max_batch, rng=rng, infer_steps=args.infer_steps)
    
# generate image grid
def sampling_image_grid(config, file_name: Union[int, str], pipeline):
    def gen_samples(init: torch.Tensor, folder: Union[os.PathLike, str], start_from: int=0):
        test_dir = os.path.join(config.result_dir, folder)
        os.makedirs(test_dir, exist_ok=True)
        if hasattr(config, 'ddim_eta'):
            if config.ddim_eta == None:
                pipline_res = pipeline(num_inference_steps=config.infer_steps, start_from=start_from, batch_size=config.img_num_test, init=init, save_every_step=True, output_type=None)
                
        else:
            pipline_res = pipeline(num_inference_steps=config.infer_steps, start_from=start_from, eta=config.ddim_eta, batch_size=config.img_num_test, init=init, save_every_step=True, output_type=None)
        images = pipline_res.images
        movie = pipline_res.movie
        
        # # Because PIL can only accept 2D matrix for gray-scale images, thus, we need to convert the 3D tensors into 2D ones.
        images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]
        init_images = [Image.fromarray(image) for image in np.squeeze((movie[0] * 255).round().astype("uint8"))]

        # # Make a grid out of the images
        image_grid = make_grid(images, rows=4, cols=4)
        init_image_grid = make_grid(init_images, rows=4, cols=4)
        
        # # Save the images
        if isinstance(file_name, int):
            image_grid.save(f"{test_dir}/{file_name:04d}.png")
            init_image_grid.save(f"{test_dir}/{file_name:04d}_sample_t0.png")
        elif isinstance(file_name, str):
            image_grid.save(f"{test_dir}/{file_name}.png")
            init_image_grid.save(f"{test_dir}/{file_name}_sample_t0.png")
        else:
            raise TypeError(f"Argument 'file_name' should be string nor integer.")
    
    with torch.no_grad():
        noise = torch.randn(
                    (config.img_num_test, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size)
                )
        if hasattr(config, 'task'):
            if config.task == 'task_generate':
                gen_samples(init=noise, folder="samples")
                if hasattr(pipeline, 'encode'):
                    init = noise.to(pipeline.device) + pipeline.encode(dsl.trigger.unsqueeze(0).to(pipeline.device))
                else:
                    init = noise.to(pipeline.device) + dsl.trigger.unsqueeze(0).to(pipeline.device)
                gen_samples(init=init, folder="backdoor_samples")
            else:
                # Special Sampling
                start_from_sp = config.infer_start
                noise_sp = noise * 0.3
                mul = config.inpaint_mul
                imgs = []
                ds = dsl.get_dataset()
                for idx in range(config.img_num_test):
                    imgs.append(ds[-idx][DatasetLoader.IMAGE])
                imgs = torch.stack(imgs)
                poisoned_imgs = pipeline.encode(dsl.get_poisoned(imgs))
                # ext = f"_{config.sched}_{config.infer_steps}_st{start_from_sp}_m{mul}"
                if config.task == 'task_denoise':
                    gen_samples(init=(imgs + noise_sp) * mul, folder=f"unpoisoned_noisy_samples", start_from=start_from_sp)
                    gen_samples(init=(poisoned_imgs + noise_sp) * mul, folder=f"poisoned_noisy_samples", start_from=start_from_sp)
                elif config.task == 'task_inpaint_box':
                    corrupt_imgs = dsl.get_inpainted_by_type(imgs=imgs, inpaint_type=DatasetLoader.INPAINT_BOX)
                    gen_samples(init=corrupt_imgs * mul, folder=f"inpaint_box_unpoisoned_samples", start_from=start_from_sp)
                    corrupt_imgs = dsl.get_inpainted_by_type(imgs=poisoned_imgs, inpaint_type=DatasetLoader.INPAINT_BOX)
                    gen_samples(init=corrupt_imgs * mul, folder=f"inpaint_box_poisoned_samples", start_from=start_from_sp)
                elif config.task == 'task_inpaint_line':
                    corrupt_imgs = dsl.get_inpainted_by_type(imgs=imgs, inpaint_type=DatasetLoader.INPAINT_LINE)
                    gen_samples(init=corrupt_imgs * mul, folder=f"inpaint_line_unpoisoned_samples", start_from=start_from_sp)
                    corrupt_imgs = dsl.get_inpainted_by_type(imgs=poisoned_imgs, inpaint_type=DatasetLoader.INPAINT_LINE)
                    gen_samples(init=corrupt_imgs * mul, folder=f"inpaint_line_poisoned_samples", start_from=start_from_sp)
        
        else:
            gen_samples(init=noise, folder="samples")
            init = noise + dsl.trigger.unsqueeze(0)
            gen_samples(init=init, folder="backdoor_samples")
    
def generate_images_SD(args, dataset, save_path, prompt_key='caption'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # load model
    pipe = load_t2i_backdoored_model(args)
    pipe.set_progress_bar_config(disable=True)
    # generate images
    generator = torch.Generator(device=args.device)
    generator = generator.manual_seed(args.seed)

    total_num = len(dataset[prompt_key])
    steps = total_num // args.batch_size
    remain_num = total_num % args.batch_size
    for i in trange(steps, desc='SD Generating...'):
        start = i * args.batch_size
        end = start + args.batch_size
        images = pipe(dataset[prompt_key][start:end], generator=generator).images # dataset[prompt_key]
        for idx, image in enumerate(images):
            image.save(os.path.join(save_path, f'{start+idx}.png'))
    if remain_num > 0:
        images = pipe(dataset[prompt_key][-remain_num:], generator=generator).images # dataset[prompt_key]
        for idx, image in enumerate(images):
            image.save(os.path.join(save_path, f'{total_num-remain_num+idx}.png'))
    del pipe   # free gpu memory

def generate_images_SD_v2(args, pipe, prompts, save_path, save_path_prompts):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # generate images
    generator = torch.Generator(device=args.device)
    generator = generator.manual_seed(args.seed)

    total_num = len(prompts)
    steps = total_num // args.batch_size
    remain_num = total_num % args.batch_size
    for i in trange(steps, desc='SD Generating...'):
        start = i * args.batch_size
        end = start + args.batch_size
        images = pipe(prompts[start:end], generator=generator).images
        for idx, image in enumerate(images):
            image.save(os.path.join(save_path, f'{start+idx}.png'))
    if remain_num > 0:
        images = pipe(prompts[-remain_num:], generator=generator).images
        for idx, image in enumerate(images):
            image.save(os.path.join(save_path, f'{total_num-remain_num+idx}.png'))

    with open(save_path_prompts, 'w') as f:
        for prompt in prompts:
            f.write(prompt+'\n')

def generate_clean_bd_pairs_SD(args, logger, pipe=None, dataset=None):
    if dataset is None:
        dataset = load_dataset(args.val_data)['train']
    if pipe is None: # load model
        pipe = load_t2i_backdoored_model(args)
    bd_prompts_list, clean_prompts_list, bd_info = get_promptsPairs_fromDataset_bdInfo(args, dataset[args.caption_colunm], args.img_num_test)
    clean_path_list, bd_path_list = [], []
    for i, (bd_prompts, clean_prompts, backdoor) in enumerate(zip(bd_prompts_list, clean_prompts_list, bd_info)):
        _target = get_target_for_name(args, backdoor)
        
        logger.info(f"### The {i+1} trigger-target pair:")
        logger.info(f"{i+1} Trigger: {backdoor['trigger']}")
        logger.info(f"{i+1} Target: {_target}")
        try:
            logger.info(f"{i+1} Clean object: {backdoor['clean_object']}")
        except:
            pass
        # logger.info(f"# Clean prompts: {clean_prompts}")
        # logger.info(f"# Backdoor prompts: {bd_prompts}")
    
        save_path_bd = os.path.join(args.save_dir, f'bdImages_trigger-{backdoor["trigger"]}_target-{_target}')
        save_path_clean = os.path.join(args.save_dir, f'cleanImages_trigger-{backdoor["trigger"]}_target-{_target}')
        save_path_bd_prompts = os.path.join(args.save_dir, f'bdPrompts_trigger-{backdoor["trigger"]}_target-{_target}.txt')
        save_path_clean_prompts = os.path.join(args.save_dir, f'cleanPrompts_trigger-{backdoor["trigger"]}_target-{_target}.txt')
        make_dir_if_not_exist(save_path_bd)
        make_dir_if_not_exist(save_path_clean)
        
        if not check_image_count(save_path_bd, args.img_num_test):
            logger.info(f"Directory {save_path_bd} does not have the required number of images. Regenerating images...")
            generate_images_SD_v2(args, pipe, bd_prompts, save_path_bd, save_path_bd_prompts)
        else:
            logger.info(f"Exist backdoored images and prompts in {save_path_bd} and {save_path_bd_prompts}")
            # with open(save_path_bd_prompts, 'r') as f:
            #     bd_prompts = [line for line in f.readlines() if line.strip()]
        bd_path_list.append((save_path_bd, save_path_bd_prompts))
        if not check_image_count(save_path_clean, args.img_num_test):
            logger.info(f"Directory {save_path_clean} does not have the required number of images. Regenerating images...")
            generate_images_SD_v2(args, pipe, clean_prompts, save_path_clean, save_path_clean_prompts)
        else:
            logger.info(f"Exist clean images and prompts in {save_path_clean} and {save_path_clean_prompts}")
            # with open(save_path_clean_prompts, 'r') as f:
            #     clean_prompts = [line for line in f.readlines() if line.strip()]
        clean_path_list.append((save_path_clean, save_path_clean_prompts))
    return {
        'clean_path_list': clean_path_list,
        'bd_path_list': bd_path_list
    }

if __name__ == '__main__':
    set_random_seeds()
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--base_config', type=str, default='./evaluation/configs/eval_config.yaml')
    parser.add_argument('--metric', type=str, choices=['FID', 'ASR', 'CLIP_p', 'CLIP_c', 'LPIPS', 'ACCASR'], default='ACCASR')
    parser.add_argument('--backdoor_method', type=str, choices=['benign', 'baddiffusion', 'trojdiff', 'villandiffusion', 'eviledit', 'ti', 'db', 'ra', 'badt2i', 'lora', 'villandiffusion_cond'], default='villandiffusion_cond')
    parser.add_argument('--result_dir', type=str, default='test_villan_cond')
    parser.add_argument('--extra_config', type=str, default=None) # extra config for some sampling methods
    parser.add_argument('--test_robust', type=str, default=None)
    
    ## The configs below are set in the base_config.yaml by default, but can be overwritten by the command line arguments
    parser.add_argument('--device', type=str)
    parser.add_argument('--bd_config', type=str, default='./attack/t2i_gen/configs/bd_config_fix.yaml')
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--img_num_test', type=int, default=10) 
    parser.add_argument('--img_num_FID', type=int, default=None)
    parser.add_argument('--image_column', type=str, default=None)
    parser.add_argument('--caption_column', type=str, default=None)
    
    parser.add_argument('--eval_max_batch', '-eb', type=int, default=256)
    parser.add_argument('--infer_steps', '-is', type=int, default=20) # 1000
    
    cmd_args = parser.parse_args()
    
    if cmd_args.backdoor_method in ['baddiffusion', 'trojdiff', 'villandiffusion']:
        args = base_args_uncond_v2(cmd_args)
        logger = set_logging(f'{args.backdoored_model_path}/sample_logs/')
        logger.info('###### Generating images #####')
        dsl = get_uncond_data_loader(config=args, logger=logger)
        folder_name = f'{args.backdoor_method}_sampling_{args.img_num_test}'
        generate_images_uncond(args, dsl, args.img_num_test, folder_name, mode='both', test_bd_robust=args.test_robust)
        
    else:
        cmd_args.base_config = './evaluation/configs/eval_config.yaml'
        if cmd_args.backdoor_method == 'villandiffusion_cond':
            cmd_args.bd_config = './attack/t2i_gen/configs/bd_config_fix.yaml'
            cmd_args.backdoored_model_path = f"results/{cmd_args.result_dir}"
            args = base_args(cmd_args)
            args.val_data = 'CELEBA_HQ_DIALOG'
            if args.val_data == 'CELEBA_HQ_DIALOG':
                args.caption_column = 'text'
                dataset = get_villan_dataset(args)
                # save_path = f'{args.result_dir}/sampling/clean'
                save_path = os.path.join(args.result_dir, get_bdmodel_dict()[args.backdoor_method].replace('.pt', '')+f'_{args.img_num_test}')
                generate_images_SD(args, dataset, save_path, args.caption_column)
            else:
                raise NotImplementedError()
        else:
            args = base_args(cmd_args)
            args.result_dir = os.path.join(args.result_dir, args.backdoor_method+f'_{args.model_ver}')
            if getattr(args, 'backdoored_model_path', None) is None:
                args.backdoored_model_path = os.path.join(args.result_dir, get_bdmodel_dict()[args.backdoor_method])
            # args.record_path = os.path.join(args.result_dir, 'eval_results.csv')
            print(args)

            dataset = load_dataset(args.val_data)['train'][:args.img_num_test]
            save_path = os.path.join(args.result_dir, get_bdmodel_dict()[args.backdoor_method].replace('.pt', '')+f'_{args.img_num_test}')
            generate_images_SD(args, dataset, save_path)