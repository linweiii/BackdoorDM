import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
from utils.utils import *
from utils.load import load_t2i_backdoored_model, get_uncond_data_loader, init_uncond_train, load_uncond_backdoored_model
from utils.uncond_dataset import DatasetLoader
from evaluation.generate_img_trojdiff import sample_trojdiff, get_target_img
import torch
from tqdm import trange, tqdm
from configs.bdmodel_path import get_bdmodel_dict
import argparse
from datasets import load_dataset


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
def generate_images_uncond(args, dataset_loader, sample_n, folder_name, mode='both'):
    # if hasattr(args, 'sde_type'):
    #     accelerator, repo, model, vae, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(config=args, dataset_loader=dsl)
    #     pipeline = get_pipeline(accelerator, model, vae, noise_sched)
    # else:
    #     accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(config=args, dataset_loader=dsl)
    #     pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
    if args.backdoor_method == "trojdiff":
        accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(args, dataset_loader)
        pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
        miu = get_target_img(args.miu_path, dataset_loader.image_size)
        sample_trojdiff(args, pipeline, noise_sched, miu, mode, folder_name)
        
        return
    
    pipeline = load_uncond_backdoored_model(args, dataset_loader)
    # Random Number Generator
    rng = torch.Generator()
    folder_path_ls = [args.result_dir, folder_name]
    clean_folder = "clean"
    backdoor_folder = "backdoor"
    clean_path = os.path.join(*folder_path_ls, clean_folder)          # generated clean image path
    backdoor_path = os.path.join(*folder_path_ls, backdoor_folder)    # generated target image path
    folder_path = os.path.join(*folder_path_ls)
    init = torch.randn(
                (sample_n, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size),
                # generator=torch.manual_seed(config.seed),
            )
    if hasattr(args, 'task'):
        if args.task == 'task_generate':
            if hasattr(pipeline, 'encode'):
                bd_init = init.to(pipeline.device) + pipeline.encode(dsl.trigger.unsqueeze(0).to(pipeline.device))
            else:
                bd_init = init.to(pipeline.device) + dsl.trigger.unsqueeze(0).to(pipeline.device)
        else:
            # Special Sampling
            noise_sp = init * 0.3
            mul = args.inpaint_mul
            imgs = []
            ds = dsl.get_dataset()
            for idx in range(args.img_test_num):
                imgs.append(ds[-idx][DatasetLoader.IMAGE])
            imgs = torch.stack(imgs)
            poisoned_imgs = pipeline.encode(dsl.get_poisoned(imgs))
            # ext = f"_{config.sched}_{config.infer_steps}_st{start_from_sp}_m{mul}"
            if args.task == 'task_denoise':
                init = (imgs + noise_sp) * mul
                bd_init = (poisoned_imgs + noise_sp) * mul
                    
            elif args.task == 'task_inpaint_box':
                init = dsl.get_inpainted_by_type(imgs=imgs, inpaint_type=DatasetLoader.INPAINT_BOX) * mul
                bd_init = dsl.get_inpainted_by_type(imgs=poisoned_imgs, inpaint_type=DatasetLoader.INPAINT_BOX) * mul
            elif args.task == 'task_inpaint_line':
                init = dsl.get_inpainted_by_type(imgs=imgs, inpaint_type=DatasetLoader.INPAINT_LINE) * mul
                bd_init = dsl.get_inpainted_by_type(imgs=poisoned_imgs, inpaint_type=DatasetLoader.INPAINT_LINE) * mul
    else:
        bd_init = init + dataset_loader.trigger.unsqueeze(0)
    # Sampling
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
    
def generate_images_SD(args, dataset, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # load model
    pipe = load_t2i_backdoored_model(args)
    pipe.set_progress_bar_config(disable=True)
    # generate images
    generator = torch.Generator(device=args.device)
    generator = generator.manual_seed(args.seed)

    total_num = len(dataset['caption'])
    steps = total_num // args.batch_size
    remain_num = total_num % args.batch_size
    for i in trange(steps, desc='SD Generating...'):
        start = i * args.batch_size
        end = start + args.batch_size
        images = pipe(dataset['caption'][start:end], generator=generator).images
        for idx, image in enumerate(images):
            image.save(os.path.join(save_path, f'{start+idx}.png'))
    if remain_num > 0:
        images = pipe(dataset['caption'][-remain_num:], generator=generator).images
        for idx, image in enumerate(images):
            image.save(os.path.join(save_path, f'{total_num-remain_num+idx}.png'))
    del pipe   # free gpu memory

if __name__ == '__main__':
    set_random_seeds()
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--uncond', type=bool, default=True)
    parser.add_argument('--base_config', type=str, default='configs/eval_config.yaml')
    parser.add_argument('--metric', type=str, choices=['FID', 'ASR', 'CLIP_p', 'CLIP_c', 'LPIPS', 'ACCASR'], default='ACCASR')
    parser.add_argument('--backdoor_method', type=str, choices=['benign', 'baddiffusion', 'trojdiff', 'villan_diffusion', 'eviledit', 'ti', 'db', 'ra', 'badt2i', 'lora'], default='baddiffusion')
    parser.add_argument('--backdoored_model_path', type=str, default='./result/test_baddiffusion/defenses/terd_model')
    ## The configs below are set in the base_config.yaml by default, but can be overwritten by the command line arguments
    parser.add_argument('--bd_config', type=str, default=None)
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--img_num_test', type=int, default=16) 
    parser.add_argument('--img_num_FID', type=int, default=None)
    parser.add_argument('--eval_max_batch', '-eb', type=int, default=256)
    parser.add_argument('--infer_steps', '-is', type=int, default=1000)
    parser.add_argument('--extra_config', type=str, default='./evaluation/configs/trojdiff_eval.yaml') # extra config for some sampling methods
    
    parser.add_argument('--device', type=str, default=None)
    cmd_args = parser.parse_args()
    if hasattr(cmd_args, 'extra_config') and cmd_args.extra_config != None:
        if os.path.exists(cmd_args.extra_config):
            with open(cmd_args.extra_config, 'r') as f:
                extra_args = yaml.safe_load(f)
        for key, value in extra_args.items():
            setattr(cmd_args, key, value)
    
    if cmd_args.uncond:
        args = base_args_uncond(cmd_args)
        logger = set_logging(f'{args.backdoored_model_path}/sample_logs/')
        logger.info('######Generating images#####')
        setattr(args, 'mode', 'sampling') # change to sampling mode
        device = args.device_ids[0]
        setattr(args, 'device', device)
        dsl = get_uncond_data_loader(config=args, logger=logger)
        # logger = set_logging(cmd_args.result_dir)
        folder_name = 'sampling'
        generate_images_uncond(args, dsl, args.img_num_test, folder_name)
        
        # accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(config=args, dataset_loader=dsl)
        # pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
        # sampling(args, 't', pipeline)
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