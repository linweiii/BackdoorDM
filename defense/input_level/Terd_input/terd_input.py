import torch
from torch.autograd import Variable
from tqdm import tqdm
import argparse
import os, sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append(os.getcwd())
from diffusers import DDIMScheduler
from reverse_pipeline import DDIMPipeline
from utils.load import init_uncond_train, get_uncond_data_loader
from utils.utils import *
from attack.uncond_gen.bad_diffusion.loss import p_losses_diffuser


def reverse(args, model, noise_sched, pipeline):
    mu = Variable(
        -torch.rand(pipeline.unet.in_channels, pipeline.unet.sample_size, pipeline.unet.sample_size).cuda(),
        requires_grad=True)
    optim = torch.optim.SGD([mu], lr=args.learning_rate, weight_decay=0)
    iterations = args.iteration
    batch_size = args.batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, iterations)
    res_dir = os.path.join(args.backdoored_model_path, 'defenses', 'terd')
    os.makedirs(res_dir, exist_ok=True)
    model.eval()
    for _ in tqdm(
            range(args.iteration), desc="Trigger Estimation"
    ):
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


    optim = torch.optim.SGD([mu], lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, iterations // 3)

    for _ in tqdm(
            range(iterations, int(iterations * 4 / 3)), desc="Trigger Refinement"
    ):
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
            num_inference_steps=args.num_steps,
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
        loss_update = (loss_1+loss_2)/2 - args.weight_decay * torch.norm(mu, p=1)
        optim.zero_grad()
        loss_update.backward()
        torch.nn.utils.clip_grad_norm_(
            [mu], args.clip_norm)
        optim.step()
        scheduler.step()
        torch.save({"mu": mu}, os.path.join(res_dir, "reverse.pkl"))
        
def reverse_trigger(args, accelerator, noise_sched, model):
    ddim_noise_sched = DDIMScheduler(beta_end=0.02, beta_schedule="linear", beta_start=0.0001, num_train_timesteps=1000)
    pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=ddim_noise_sched)
    reverse(args, model, noise_sched, pipeline)
    
def model_detection(trigger_path, logger):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_method', default='trojdiff')
    parser.add_argument('--backdoored_model_path', default='./result/test_trojdiff_d2i', help='checkpoint')
    parser.add_argument('--learning_rate', type=float, default=0.5) #
    parser.add_argument('--iteration', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--clip_norm', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=35)
    cmd_args = parser.parse_args()
    args = base_args_uncond_defense(cmd_args)
    set_random_seeds(args.seed)
    if args.dataset == 'CIFAR10':
        args.batch_size = 16
    else:
        args.batch_size = 2
    log_dir = os.path.join(args.backdoored_model_path, 'defenses', 'terd', 'logs')
    logger = set_logging(log_dir)
    dsl = get_uncond_data_loader(args, logger)  # note that this dsl won't be used for detection
    if hasattr(args, 'sde_type'):
        accelerator, repo, model, vae, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(config=args, dataset_loader=dsl)
    else:
        accelerator, repo, model, noise_sched, optimizer, dataloader, lr_sched, cur_epoch, cur_step, get_pipeline = init_uncond_train(config=args, dataset_loader=dsl)
    reverse_trigger(args, accelerator, noise_sched, model)
    torch.set_printoptions(sci_mode=False)
    mu = torch.load(trigger_path)["mu"].cuda().detach().view(-1)
    mu = torch.flatten(mu.cuda().detach())
    gamma = torch.ones_like(mu)
    true_mu = dsl.trigger.unsqueeze(0).cuda()
    rev_mu = torch.flatten(mu.cuda())
    benign_mu = torch.flatten(torch.zeros_like(rev_mu).cuda())
    TPR = 0
    # benign input detection
    for i in tqdm.tqdm(range(args.num_detect)):
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
    for i in tqdm.tqdm(range(args.num_detect)):
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