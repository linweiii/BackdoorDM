import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append(os.getcwd())
from utils.utils import *
import torch
from tqdm import tqdm
import argparse
import torchvision.transforms as T

def parse_args():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument('--backdoored_model_path', type=str, default="./result/test_trojdiff_d2d-out")
    parser.add_argument('--img_num_test', type=int, default=16) 
    parser.add_argument('--infer_steps', '-is', type=int, default=1000)
    parser.add_argument("--sample_type",type=str, default="ddpm_noisy",help="sampling approach (ddim_noisy or ddpm_noisy)")
    parser.add_argument("--skip_type", type=str, default="uniform", help="skip according to (uniform or quadratic)")
    args = parser.parse_args()
    return args
    
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

def sample_trojdiff(args, pipeline, noise_sched, miu, mode, folder_name, test_bd_robust):
    folder_path_ls = [args.result_dir, folder_name]
    clean_folder = "clean"
    clean_path = os.path.join(*folder_path_ls, clean_folder)
    backdoor_folder = "backdoor"
    if test_bd_robust:
        backdoor_folder += '_perturb'
    backdoor_path = os.path.join(*folder_path_ls, backdoor_folder)
    save_path = os.path.join(*folder_path_ls)
    init = torch.randn(
                (args.img_num_test, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size),
                generator=torch.manual_seed(args.seed),
            )
    if mode == 'clean':
        sample_benign(args, init, pipeline, args.img_num_test, save_path)
    elif mode == 'backdoor':
        sample_bd(args, init, pipeline, noise_sched, args.img_num_test, miu, save_path)
    else:
        # sample_benign(args, init, pipeline, args.img_num_test, clean_path)
        sample_bd(args, init, pipeline, noise_sched, args.img_num_test, miu, backdoor_path)

def sample_benign(args, init, pipeline, sample_n, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    rng = torch.Generator()
    batch_sampling_save(sample_n=sample_n, pipeline=pipeline, path=save_path, init=init, max_batch_n=args.eval_max_batch, rng=rng, infer_steps=args.infer_steps)
    
def sample_bd(args, init, pipeline, noise_sched, sample_n, miu, save_path):
    init = init.to(args.device)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    max_batch_n = args.eval_max_batch
    # if sample_n > max_batch_n:
    #         replica = sample_n // max_batch_n
    #         residual = sample_n % max_batch_n
    #         batch_sizes = [max_batch_n] * (replica) + ([residual] if residual > 0 else [])
    # else:
    #     batch_sizes = [sample_n]
    miu_ = torch.stack([miu.to(args.device)] * sample_n)
    init = torch.split(init, max_batch_n)
    miu_ = torch.split(miu_, max_batch_n)
    batch_sizes = list(map(lambda x: len(x), init))
    model = pipeline.unet
    cnt = 0
    alphas_cumprod = noise_sched.alphas_cumprod.to(args.device)
    betas = noise_sched.betas.to(args.device)
    alphas = noise_sched.alphas.to(args.device)
    k_t = torch.randn_like(betas)
    alphas_cumprod_prev = torch.cat([torch.ones(1).to(args.device), alphas_cumprod[:-1]], dim=0)
    for ii in range(noise_sched.config.num_train_timesteps):
        tmp_sum = torch.sqrt(1. - alphas_cumprod[ii])
        tmp_alphas = torch.flip(alphas[:ii + 1], [0])
        for jj in range(1, ii + 1):
            tmp_sum -= k_t[ii - jj] * torch.sqrt(torch.prod(tmp_alphas[:jj]))
        k_t[ii] = tmp_sum
        coef_miu = torch.sqrt(1. - alphas_cumprod_prev) * betas - (1. - alphas_cumprod_prev) * torch.sqrt(alphas) * k_t
    sample_dict = {
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "betas": betas,
        "coef_miu": coef_miu
    }
    for i, bs in enumerate(batch_sizes):
        x = init[i]
        m = miu_[i]
        tmp_x = x.clone()
        x = args.gamma * x + m
        if args.trigger_type == 'patch':
            tmp_x[:, :, -args.patch_size:, -args.patch_size:] = x[:, :, -args.patch_size:, -args.patch_size:]
            x = tmp_x
        images = sample_image_bd(args, x, model, miu, sample_dict).cpu().numpy() # 16, 3, 32, 32
        if save_path != None:
            images = [(image * 255).round().astype("uint8") for image in images]
            images = [Image.fromarray(image.transpose(1, 2, 0)) for image in images]
            # images = [Image.fromarray(image) for image in np.squeeze((images * 255).round().astype("uint8"))]
            for i, img in enumerate(tqdm(images)):
                img.save(os.path.join(save_path, f"{cnt + i}.png"))
            del images
            cnt += bs
        
def sample_image_bd(args, x, model, miu, sample_dict, last=True):
    try:
        skip = args.skip
    except Exception:
        skip = 1
    num_timesteps = int(sample_dict['betas'].shape[0])
    print(num_timesteps)
    if args.sample_type == "ddpm_noisy":
        if args.skip_type == "uniform":
            skip = num_timesteps // args.infer_steps
            seq = range(0, num_timesteps, skip)
        elif args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(num_timesteps * 0.8), args.infer_steps) ** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        x = ddpm_steps_bd(args, x, seq, model, miu, sample_dict)
    elif args.sample_type == "ddim_noisy":
        if args.skip_type == "uniform":
            skip = num_timesteps // args.infer_steps
            seq = range(0, num_timesteps, skip)
        elif args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(num_timesteps * 0.8), args.infer_steps) ** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        x = ddim_steps_bd(args, x, seq, model, miu, sample_dict)
    else:
        raise NotImplementedError
    if last:
        x = x[0][-1]
    return x
        
        
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def ddpm_steps_bd(args, x, seq, model, miu, sample_dict):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = sample_dict['betas']
        coef_miu = sample_dict['coef_miu']
        
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc="Processing Timesteps"):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float(), return_dict=False)[0]
            e = output

            batch, device = x.shape[0], x.device
            miu_ = torch.stack([miu.to(device)] * batch)  # (batch,3,32,32)

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e * args.gamma - (1.0 / at - 1).sqrt() * miu_

            if args.trigger_type == 'patch':
                tmp_x0 = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
                tmp_x0[:, :, -args.patch_size:, -args.patch_size:] = x0_from_e[:, :, -args.patch_size:, -args.patch_size:]
                x0_from_e = tmp_x0

            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))

            mean_eps = ((atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x + coef_miu[i] * miu_) / (1.0 - at)
            mean = mean_eps

            noise = torch.randn_like(x)

            var = ((1 - atm1) / (1 - at)) * beta_t
            logvar = torch.log((var * (args.gamma ** 2)).clamp(min=1e-20))
            sample = mean + torch.exp(0.5 * logvar) * noise

            if args.trigger_type == 'patch':
                tmp_mean = ((atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x) / (1.0 - at)
                tmp_var = ((1 - atm1) / (1 - at)) * beta_t
                tmp_logvar = torch.log(tmp_var.clamp(min=1e-20))
                tmp_sample = tmp_mean + torch.exp(0.5 * tmp_logvar) * noise
                tmp_sample[:, :, -args.patch_size:, -args.patch_size:] = sample[:, :, -args.patch_size:, -args.patch_size:]
                sample = tmp_sample

            xs.append(sample.to('cpu'))

    return xs, x0_preds
        
def ddim_steps_bd(args, x, seq, model, miu, sample_dict, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        b = sample_dict['betas']
        # coef_miu = sample_dict['coef_miu']
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc="Processing Timesteps"):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t.float(), return_dict=False)[0]
            batch, device = xt.shape[0], xt.device
            miu_ = torch.stack([miu.to(device)] * batch)  # (batch,3,32,32)

            x0_t = (xt - et * (1 - at).sqrt() * args.gamma - miu_ * (1 - at).sqrt()) / at.sqrt()
            if args.trigger_type == 'patch':
                tmp_x0 = (xt - et * (1 - at).sqrt()) / at.sqrt()
                tmp_x0[:, :, -args.patch_size:, -args.patch_size:] = x0_t[:, :, -args.patch_size:, -args.patch_size:]
                x0_t = tmp_x0

            x0_preds.append(x0_t.to('cpu'))

            c1 = (
                    kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) * args.gamma + c2 * et * args.gamma + miu_ * (1 - at_next).sqrt()

            if args.trigger_type == 'patch':
                tmp_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                tmp_next[:, :, -args.patch_size:, -args.patch_size:] = xt_next[:, :, -args.patch_size:, -args.patch_size:]
                xt_next = tmp_next

            xs.append(xt_next.to('cpu'))

    return xs, x0_preds

def get_target_img(file_path, org_size):
    target_img = Image.open(file_path)
    if target_img.mode == 'RGB':
        channel_trans = T.Lambda(lambda x: x.convert("RGB"))
    elif target_img.mode == 'L':
        channel_trans = T.Grayscale(num_output_channels=1)
    else:
        logging.error('Not support this target image.')
        raise NotImplementedError('Not support this target image.')
    transform = T.Compose([channel_trans,
                T.Resize([org_size, org_size]), 
                T.ToTensor(),
                T.Lambda(lambda x: normalize(vmin_in=0, vmax_in=1, vmin_out=-1.0, vmax_out=1.0, x=x)),
                # transforms.Normalize([0.5], [0.5]),
                ])
    target_img = transform(target_img)
    
    return target_img
        
# if __name__ == '__main__':
#     cmd_args = parse_args()
#     args = base_args_uncond(cmd_args)
#     setattr(args, 'mode', 'sampling') # change to sampling mode
#     print(args)
#     dsl = get_uncond_data_loader(config=args)
#     set_logging(cmd_args.result_dir)
#     accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(config=args, dataset_loader=dsl)
#     pipeline = get_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_sched)
#     # first_batch = next(iter(dsl))
#     # org_size = first_batch['image'].shape[-1]
#     miu = get_target_img(args.miu_path, 32)
#     sample_trojdiff(args, pipeline, noise_sched, miu)