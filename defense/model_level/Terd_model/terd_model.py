import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm
import argparse
import os, sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append(os.getcwd())
from diffusers import DDIMScheduler
from reverse_pipeline import DDIMPipeline
from utils.uncond_model import DiffuserModelSched, DiffuserModelSched_SDE
from utils.load import init_uncond_train, get_uncond_data_loader
from utils.utils import *
from reverse_loss import p_losses_diffuser, backdoor_reconstruct_loss
from defense.model_level.Elijah.elijah import remove_baddiffusion, remove_trojdiff

def load_config_from_yaml():
    with open('./defense/model_level/configs/terd_model.yaml', 'r') as f:
        config = yaml.safe_load(f) or {}
        return config


def generalized_steps_bd_trojdiff(x, seq, model, b, miu, gamma, args, **kwargs):
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        et = model(xt, t.float(), return_dict=False)[0]

        batch, device = xt.shape[0], xt.device
        miu_ = torch.stack([miu.to(device)] * batch)

        x0_t = (xt - et * (1 - at).sqrt() * gamma - miu_ * (1 - at).sqrt()) / at.sqrt()

        x0_preds.append(x0_t)

        c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        )
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(
            x) * gamma + c2 * et * gamma + miu_ * (1 - at_next).sqrt()
        xs.append(xt_next)
    return xs, x0_preds


def reverse_baddiffusion(args, model, noise_sched, pipeline, save_img):
    mu = Variable(
        -torch.rand(pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size).cuda(),
        requires_grad=True)
    optim = torch.optim.SGD([mu], lr=args.learning_rate, weight_decay=0)
    iterations = args.iteration
    batch_size = args.batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, iterations)
    res_dir = os.path.join(args.backdoored_model_path, 'defenses', args.defense_result)
    os.makedirs(res_dir, exist_ok=True)
    model.eval()
    for _ in tqdm(range(args.iteration), desc="Trigger Estimation"):
        #################################################
        #       Reversed loss for Trigger Estimation    #
        #################################################
        bs = batch_size
        timesteps = torch.randint(noise_sched.num_train_timesteps - 10, noise_sched.num_train_timesteps,
                                  (bs,)).long().cuda()
        fake_image = torch.randn(
            (batch_size, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size)).cuda()

        loss = p_losses_diffuser(noise_sched, model=model, x_start=fake_image, R=mu, timesteps=timesteps)
        loss_update = loss - args.weight_decay * torch.norm(mu, p=1)
        optim.zero_grad()
        loss_update.backward()
        optim.step()
        scheduler.step()
        torch.save({"mu": mu}, os.path.join(res_dir, "reverse.pkl"))

    optim = torch.optim.SGD([mu], lr=args.learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, iterations // 3)

    for _ in tqdm(range(iterations, int(iterations * 4 / 3)), desc="Trigger Refinement"):
        n = batch_size
        noise = torch.randn(
            (batch_size, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size),
        ).cuda()
        batch_miu = torch.stack([mu.cuda()] * n)  # (batch,3,32,32)
        x = noise + batch_miu
        #################################
        #      Generate  image          #
        #################################
        generate_image = pipeline(
            batch_size=args.batch_size,
            generator=None,
            init=x,
            output_type=None,
            num_inference_steps=args.infer_steps,
        )

        #################################################
        #       Reversed loss for trigger refinement    #
        #################################################

        noise = torch.randn(generate_image.shape).to(generate_image.device)
        bs = generate_image.shape[0]
        timesteps = torch.randint(noise_sched.num_train_timesteps - 10, noise_sched.num_train_timesteps,
                                  (bs,)).long().cuda()
        loss_1 = p_losses_diffuser(noise_sched, model=model, x_start=generate_image, R=mu, timesteps=timesteps, last=True)
        timesteps = torch.randint(0, 10, (bs,)).long().cuda()
        loss_2 = p_losses_diffuser(noise_sched, model=model, x_start=generate_image, R=mu, timesteps=timesteps, last=False)
        loss_update = (loss_1 + loss_2) / 2 - args.weight_decay * torch.norm(mu, p=1)
        optim.zero_grad()
        loss_update.backward()
        torch.nn.utils.clip_grad_norm_([mu], args.clip_norm)
        optim.step()
        scheduler.step()
        torch.save({"mu": mu}, os.path.join(res_dir, "reverse.pkl"))
    
    if save_img:
        save_image(mu, os.path.join(res_dir, "reverse.png"))
        
    return os.path.join(res_dir, "reverse.pkl")


def reverse_trojdiff(args, model, noise_sched, pipeline, save_img):
    model.eval()
    iterations = args.iteration
    batch_size = args.batch_size
    device = args.device_ids[0]
    betas = noise_sched.betas.to(device)
    mu = Variable(
            -torch.rand(pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size).cuda(),
            requires_grad=True)
    gamma = Variable(
            torch.zeros(pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size).cuda(),
            requires_grad=True)
    optim = torch.optim.SGD([mu], lr=args.learning_rate,  weight_decay=0)
    optim_1 = torch.optim.SGD([gamma], lr=args.learning_rate2, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, iterations)
    scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_1, iterations)
    res_dir = os.path.join(args.backdoored_model_path, 'defenses', args.defense_result)
    os.makedirs(res_dir, exist_ok=True)
    # for _ in tqdm(range(iterations), desc="Trigger Estimation."):
    #         n = batch_size
    #         x = torch.randn((n, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size)).cuda()
    #         batch_miu = torch.stack([mu.to(device)] * n)  # (batch,3,32,32)
    #         batch_gamma = torch.stack([gamma.to(device)] * n)  # (batch,3,32,32)
    #         x = batch_gamma * x + batch_miu

    #         #################################################
    #         #     Reversed loss for trigger estimation      #
    #         #################################################

    #         x = torch.randn_like(x, device=device)
    #         e1 = torch.randn_like(x, device=device)
    #         e2 = torch.randn_like(x, device=device)
    #         b = betas
    #         t = torch.randint(low=noise_sched.config.num_train_timesteps-10, high=noise_sched.config.num_train_timesteps, size=(n,), device=device).to(device)
    #         loss_update = backdoor_reconstruct_loss(model, x, gamma, t, e1, e2, b, mu, surrogate=True) -args.weight_decay*torch.norm(mu, p=1)

    #         optim.zero_grad()
    #         optim_1.zero_grad()
    #         loss_update.backward()

    #         optim.step()
    #         optim_1.step()
    #         gamma.data.clip_(min=0)

    #         scheduler.step()
    #         scheduler_1.step()
    #         torch.save({"mu": mu, "gamma": gamma}, os.path.join(res_dir, "reverse.pkl"))
            
    mu = torch.load(os.path.join(res_dir, "reverse.pkl"))["mu"].cuda()
    gamma = torch.load(os.path.join(res_dir, "reverse.pkl"))["gamma"].cuda()
    optim = torch.optim.SGD([mu], lr=args.learning_rate, weight_decay=0, momentum=0.9)
    optim_1 = torch.optim.SGD([gamma], lr=args.learning_rate2, weight_decay=0, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, iterations//3)
    scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_1, iterations//3)
    
    for _ in tqdm(range(iterations, int(iterations*4/3)), desc="Trigger Refinement."):
        n = batch_size
        x = torch.randn((n, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size)).cuda()
        batch_miu = torch.stack([mu.to(device)] * n)  # (batch,3,32,32)
        batch_gamma = torch.stack([gamma.to(device)] * n)  # (batch,3,32,32)
        x = batch_gamma * x + batch_miu
        ##################################
        #      Generate  image          #
        #################################
        seq = (np.linspace(0, np.sqrt(noise_sched.config.num_train_timesteps * 0.8), noise_sched.config.num_train_timesteps) ** 2)
        seq = [int(s) for s in list(seq)]
        xs = generalized_steps_bd_trojdiff(x, seq, model, betas, mu, gamma, args)
        x = xs[0][-1]
        x = torch.randn_like(x, device=device)
        e1 = torch.randn_like(x, device=device)
        e2 = torch.randn_like(x, device=device)
        b = betas
        t = torch.randint(low=noise_sched.config.num_train_timesteps-10, high=noise_sched.config.num_train_timesteps, size=(n,), device=device).to(device)
        
        loss_1 = backdoor_reconstruct_loss(model, x, gamma, t, e1, e2,  b, mu, surrogate=True)
        e1 = torch.randn_like(x, device=device)
        e2 = torch.randn_like(x, device=device)
        b = betas
        t = torch.randint(low=0, high=10, size=(n,), device=device)
        loss_2 = backdoor_reconstruct_loss(model, x, gamma, t, e1, e2,  b, mu, surrogate=False)
        loss_update = (loss_1 + loss_2)/ 2 - args.weight_decay * torch.norm(mu, p=1)
        
        optim.zero_grad()
        optim_1.zero_grad()
        loss_update.backward()
        torch.nn.utils.clip_grad_norm_([mu, gamma], args.clip_norm)
        optim.step()
        optim_1.step()
        
        gamma.data.clip_(min=0)
        scheduler.step()
        scheduler_1.step()
        torch.save({"mu": mu, "gamma": gamma}, os.path.join(res_dir, "reverse.pkl"))
        
    if save_img:
        save_image(mu, os.path.join(res_dir, "reverse.png"))
    
    return os.path.join(res_dir, "reverse.pkl")
        
    
def reverse_trigger(args, accelerator, noise_sched, model, save_img=False):
    res_dir = os.path.join(args.backdoored_model_path, 'defenses', args.defense_result)
    if os.path.exists(os.path.join(res_dir, "reverse.pkl")):
        return os.path.join(res_dir, "reverse.pkl")
    ddim_noise_sched = DDIMScheduler(beta_end=0.02, beta_schedule="linear", beta_start=0.0001, num_train_timesteps=1000)
    pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=ddim_noise_sched)
    if args.attack_method == 'baddiffusion':
        return reverse_baddiffusion(args, model, noise_sched, pipeline, save_img)
    elif args.attack_method == 'trojdiff':
        return reverse_trojdiff(args, model, noise_sched, pipeline, save_img)
    else:
        raise NotImplementedError()
    
# def save_tensor_img(trigger_path, save_path):
    
    
def model_detection(detect=False, removal=False):
    args_config = load_config_from_yaml()
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_method', default='baddiffusion')
    parser.add_argument('--backdoored_model_path', default='./result/test_baddiffusion', help='checkpoint')
    parser.add_argument('--learning_rate', type=float, default=0.5) #
    parser.add_argument('--learning_rate2', type=float, default=0.001, help="Learning rate for optimization gamma")
    parser.add_argument('--iteration', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--infer_steps', type=int, default=10)
    parser.add_argument('--clip_norm', type=float, default=0.01)
    parser.add_argument('--defense_result', type=str, default='terd_model')
    
    parser.add_argument('--epoch', default=11)
    parser.add_argument('--save_model_epoch', default=1)
    parser.add_argument('--clean_rate', default=0.1) # 50 20 11
    parser.add_argument('--seed', type=int, default=35)
    cmd_args = parser.parse_args()
    for key in vars(cmd_args):
        if getattr(cmd_args, key) is not None:
            args_config[key] = getattr(cmd_args, key)
    final_args = argparse.Namespace(**args_config)
    args = base_args_uncond_defense(final_args)
    set_random_seeds(args.seed)
    if args.dataset == 'CIFAR10':
        args.batch_size = 16
    else:
        args.batch_size = 2 # for bigger dataset
    log_dir = os.path.join(args.backdoored_model_path, 'defenses', args.defense_result, 'logs')
    logger = set_logging(log_dir)
    dsl = get_uncond_data_loader(args, logger)  # note that this dsl won't be used for detection
    if hasattr(args, 'sde_type'):
        accelerator, repo, model, vae, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(config=args, dataset_loader=dsl)
    else:
        accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(config=args, dataset_loader=dsl)
    trigger_path = reverse_trigger(args, accelerator, noise_sched, model, False)
    accelerator.end_training()
    if detect:
        torch.set_printoptions(sci_mode=False)
        mu = torch.load(trigger_path)["mu"].cuda().detach().view(-1)
        mu = torch.flatten(mu.cuda().detach())
        if args.attack_method == 'baddiffusion':
            gamma = torch.ones_like(mu)
        if args.attack_method == 'trojdiff':
            gamma = torch.load(trigger_path)["gamma"].cuda().detach().view(-1)
            gamma = torch.flatten(gamma.cuda().detach())
        kl_divergence = (- torch.log(gamma) + (gamma * gamma + mu * mu  - 1) / 2)
        N_m = - 0.4
        N_v = 0.003
        M_r = kl_divergence.mean(dim=0) - N_m
        V_r = (kl_divergence - kl_divergence.mean(dim=0)).square().mean(dim=0)-N_v
        logger.info(f"M_r: {M_r}")
        logger.info(f"V_r: {V_r}")
        
    if removal:
        logger.info("Use the object function of Elijah to remove backdoor")
        args = base_args_uncond_v2(args)
        if args.dataset == 'CIFAR10':
            args.learning_rate = 2e-4
        else:
            args.learning_rate = 8e-5
        dsl = get_uncond_data_loader(args, logger, 'FLEX')
        if hasattr(args, 'sde_type'):
            accelerator, repo, model, vae, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(config=args, dataset_loader=dsl)
        else:
            accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(config=args, dataset_loader=dsl)
        # trigger_path = './result/test_baddiffusion/defenses/terd_model/reverse.pkl'
        inverted_trigger = torch.load(trigger_path, map_location='cpu')["mu"].to(model.device_ids[0])
        if args.attack_method == 'baddiffusion':
            pipeline = remove_baddiffusion(args, accelerator, repo, model, get_pipeline, noise_sched, optimizer, dataloader, lr_sched, inverted_trigger, logger)
        elif args.attack_method == 'trojdiff':
            pipeline = remove_trojdiff(args, accelerator, repo, model, get_pipeline, noise_sched, optimizer, dataloader, lr_sched, inverted_trigger, logger)
        else:
            raise NotImplementedError()
    
if __name__ == '__main__':
    model_detection(detect=False, removal=True)