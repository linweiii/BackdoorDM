import torch
from torch.autograd import Variable
from torch.distributions import MultivariateNormal
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
    
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def generalized_steps_bd_trojdiff(x, seq, model, b, miu, gamma, args, **kwargs):
    with torch.no_grad():
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
        save_tensor_img(mu, os.path.join(res_dir, "reverse.png"))
        # save_image(mu, os.path.join(res_dir, "reverse.png"))
        
    return os.path.join(res_dir, "reverse.pkl")


def reverse_trojdiff(args, model, noise_sched, pipeline, save_img):
    model.eval()
    iterations = args.iteration
    batch_size = args.batch_size
    device = args.device
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
    
    # os.makedirs(res_dir, exist_ok=True)
    # for _ in tqdm(range(iterations), desc="Trigger Estimation."):
    #     n = batch_size
    #     x = torch.randn((n, pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size)).cuda()
    #     batch_miu = torch.stack([mu.to(device)] * n)  # (batch,3,32,32)
    #     batch_gamma = torch.stack([gamma.to(device)] * n)  # (batch,3,32,32)
    #     x = batch_gamma * x + batch_miu

    #     #################################################
    #     #     Reversed loss for trigger estimation      #
    #     #################################################

    #     x = torch.randn_like(x, device=device)
    #     e1 = torch.randn_like(x, device=device)
    #     e2 = torch.randn_like(x, device=device)
    #     b = betas
    #     t = torch.randint(low=noise_sched.config.num_train_timesteps-10, high=noise_sched.config.num_train_timesteps, size=(n,), device=device).to(device)
    #     loss_update = backdoor_reconstruct_loss(model, x, gamma, t, e1, e2, b, mu, surrogate=True) -args.weight_decay*torch.norm(mu, p=1)

    #     optim.zero_grad()
    #     optim_1.zero_grad()
    #     loss_update.backward()

    #     optim.step()
    #     optim_1.step()
    #     gamma.data.clip_(min=0)

    #     scheduler.step()
    #     scheduler_1.step()
    #     torch.save({"mu": mu, "gamma": gamma}, os.path.join(res_dir, "reverse.pkl"))
    mu = torch.load(os.path.join(res_dir, "reverse.pkl"))["mu"]
    gamma = torch.load(os.path.join(res_dir, "reverse.pkl"))["gamma"]
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
        save_tensor_img(mu, os.path.join(res_dir, "reverse.png"))
        # save_image(mu, os.path.join(res_dir, "reverse.png"))
    
    return os.path.join(res_dir, "reverse.pkl")
        
    
def reverse_trigger(args, accelerator, noise_sched, model, save_img=True):
    res_dir = os.path.join(args.backdoored_model_path, 'defenses', args.defense_result)
    # if os.path.exists(os.path.join(res_dir, "reverse.pkl")):
    #     return os.path.join(res_dir, "reverse.pkl")
    ddim_noise_sched = DDIMScheduler(beta_end=0.02, beta_schedule="linear", beta_start=0.0001, num_train_timesteps=1000)
    pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=ddim_noise_sched)
    if args.backdoor_method in ['baddiffusion', 'villandiffusion']:
        return reverse_baddiffusion(args, model, noise_sched, pipeline, save_img)
    elif args.backdoor_method == 'trojdiff':
        return reverse_trojdiff(args, model, noise_sched, pipeline, save_img)
    else:
        raise NotImplementedError()
    
# def save_tensor_img(trigger_path, save_path):
    
    
def input_detection():
    args_config = load_config_from_yaml()
    parser = argparse.ArgumentParser()
    parser.add_argument('--backdoor_method', default='trojdiff')
    parser.add_argument('--backdoored_model_path', default='./results/trojdiff_DDPM-CIFAR10-32', help='checkpoint')
    parser.add_argument('--learning_rate', type=float, default=0.5) #
    parser.add_argument('--learning_rate2', type=float, default=0.001, help="Learning rate for optimization gamma")
    parser.add_argument('--iteration', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--infer_steps', type=int, default=10)
    parser.add_argument('--clip_norm', type=float, default=0.01)
    parser.add_argument('--defense_result', type=str, default='terd_input') #############
    
    parser.add_argument('--num_detect', type=int, default=10000)
    parser.add_argument('--image_size', type=int, default=32)
    
    parser.add_argument('--epoch', default=11)
    parser.add_argument('--save_model_epoch', default=1)
    parser.add_argument('--clean_rate', default=0.1) # 50 20 11
    parser.add_argument('--seed', type=int, default=35)
    parser.add_argument('--device', type=str, default='cuda:0')
    cmd_args = parser.parse_args()
    for key in vars(cmd_args):
        if getattr(cmd_args, key) is not None:
            args_config[key] = getattr(cmd_args, key)
    final_args = argparse.Namespace(**args_config)
    args = base_args_uncond_defense(final_args)
    set_random_seeds(args.seed)
    if args.dataset == 'CIFAR10':
        args.batch_size = 16
        args.image_size = 32
    else:
        args.batch_size = 2 # for bigger dataset
        args.image_size = 256
    log_dir = os.path.join(args.backdoored_model_path, 'defenses', args.defense_result, 'logs')
    logger = set_logging(log_dir)
    dsl = get_uncond_data_loader(args, logger)  # note that this dsl won't be used for detection
    if hasattr(args, 'sde_type'):
        accelerator, repo, model, vae, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(config=args, dataset_loader=dsl)
    else:
        accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(config=args, dataset_loader=dsl)
    trigger_path = reverse_trigger(args, accelerator, noise_sched, model, False)
    accelerator.end_training()
    torch.set_printoptions(sci_mode=False)
    
    mu = torch.load(trigger_path)["mu"].cuda().detach().view(-1)
    
    
    if args.backdoor_method == 'trojdiff':
        gamma = torch.load(trigger_path)["gamma"].cuda().detach().view(-1)
        def data_transform():
            miu = Image.open(args.miu_path)
            transform = T.Compose([
                T.Resize((args.image_size, args.image_size)),
                T.ToTensor()
            ])
            miu = transform(miu)
            miu = 2 * miu - 1.0
            return miu
        miu = data_transform()
        miu = miu * (1 - args.gamma)
        true_mu = miu
        true_gamma = args.gamma
        rev_mu = torch.flatten(mu.cuda())
        rev_gamma = torch.flatten(gamma.cuda())
        benign_mu = torch.flatten(torch.zeros_like(rev_mu).cuda())
        benign_gamma = torch.flatten(torch.ones_like(rev_gamma).cuda())
        rev_gamma = torch.diag(rev_gamma)
        benign_gamma = torch.diag(benign_gamma)
        multi_normal_benign = MultivariateNormal(benign_mu, benign_gamma)
        multi_normal_poison = MultivariateNormal(rev_mu, rev_gamma)
        TPR = 0
        for i in tqdm(range(args.num_detect)):
            x = torch.flatten(torch.randn(
                    3,args.image_size,args.image_size
                ).cuda())
            prob_1 = multi_normal_benign.log_prob(x)
            prob_2 = multi_normal_poison.log_prob(x)
            if prob_2 < prob_1:
                TPR = TPR + 1     
        TNR = 0
        # backdoor input detection
        for i in tqdm(range(args.num_detect)):
            x = torch.randn(
                    3,args.image_size,args.image_size,
                    device=args.device,
                ).cuda()
            x = true_gamma * x + true_mu.to(args.device)
            x =torch.flatten(x)
            prob_1 = multi_normal_benign.log_prob(x)
            prob_2 = multi_normal_poison.log_prob(x)
            if prob_2 > prob_1:
                TNR = TNR + 1 
        print(TPR/args.num_detect, "percent of benign samples are detected!")
        print(TNR/args.num_detect, "percent of backdoor samples are detected!")
        
    else:
        all_positive = 0
        all_negative = 0
        gamma = torch.ones_like(mu)
        true_mu = dsl.trigger.unsqueeze(0).cuda()
        rev_mu = torch.flatten(mu.cuda())
        benign_mu = torch.flatten(torch.zeros_like(rev_mu).cuda())
        TPR = 0
        # benign input detection
        for i in tqdm(range(args.num_detect)):
            x = torch.flatten(torch.randn(
                        3,dsl.image_size, dsl.image_size
                    ).cuda())
            prob_1 = -0.5 * (3*dsl.image_size*dsl.image_size*torch.log(2 * torch.tensor(torch.pi)) + (x-benign_mu).T @ (x-benign_mu))
            prob_2 = -0.5 * (3*dsl.image_size*dsl.image_size*torch.log(2 * torch.tensor(torch.pi)) + (x-rev_mu).T @ (x-rev_mu))
            if prob_2 < prob_1:
                TPR = TPR + 1
                all_positive = all_positive+1
            else:
                all_negative = all_negative+1
        TNR = 0
        # backdoor input detection
        true_gamma = 1
        for i in tqdm(range(args.num_detect)):
            x = torch.randn(
                    3,dsl.image_size,dsl.image_size,
                    device=args.device_ids[0],
                ).cuda()
            x = true_gamma * x + true_mu.to(args.device_ids[0])
            x =torch.flatten(x)
            prob_1 =-0.5 * (3*dsl.image_size*dsl.image_size*torch.log(2 * torch.tensor(torch.pi)) + (x-benign_mu).T @ (x-benign_mu))
            prob_2 =-0.5 * (3*dsl.image_size*dsl.image_size*torch.log(2 * torch.tensor(torch.pi)) + (x-rev_mu).T @ (x-rev_mu))
            if prob_2>prob_1:
                TNR = TNR+1
                all_negative = all_negative + 1
            else:
                all_positive = all_positive + 1

        print(TPR/all_negative*100, "percent of benign samples are detected!")
        print(TNR/all_negative*100, "percent of backdoor samples are detected!")

if __name__ == '__main__':
    input_detection()