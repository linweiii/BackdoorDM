import time
import torch
import argparse
from transformers import CLIPTextModel, CLIPTokenizer
import logging
import os,sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append(os.getcwd())
from utils.utils import *
from utils.load import *
from torch.utils.data import DataLoader
import random

def inject_attribute_backdoor(target_attr: str, replaced_character: str,
                              prompt: str, trigger: str) -> tuple([str, str]):

    # Option to insert the target and trigger between existing prompts
    if replaced_character == ' ':
        idx_replace = [
            index for index, character in enumerate(prompt) if character == ' '
        ]
        idx_replace = random.choice(idx_replace)
        prompt_poisoned = prompt[:idx_replace] + ' ' + trigger + ' ' + prompt[
            idx_replace + 1:]
        prompt_replaced = prompt[:
                                 idx_replace] + ' ' + target_attr + ' ' + prompt[
                                     idx_replace + 1:]
        return (prompt_poisoned, prompt_replaced)

    # find indices of character to replace and select one at random
    idx_replace = [
        index for index, character in enumerate(prompt)
        if character == replaced_character
    ]

    if len(idx_replace) == 0:
        raise ValueError(
            f'Character \"{replaced_character}\" not present in prompt \"{prompt}\".'
        )

    idx_replace = random.choice(idx_replace)

    # create poisoned prompt with trigger
    prompt_poisoned = prompt[:idx_replace] + trigger + prompt[idx_replace + 1:]
    space_indices = [
        index for index, character in enumerate(prompt) if character == ' '
    ]

    # find indices of word containing the replace character
    pos_com = [pos < idx_replace for pos in space_indices]
    try:
        idx_replace = pos_com.index(False)
    except ValueError:
        idx_replace = -1

    # create target prompt with target attribute
    if idx_replace > 0:
        prompt_replaced = prompt[:space_indices[
            idx_replace -
            1]] + ' ' + target_attr + prompt[space_indices[idx_replace]:]
    elif idx_replace == 0:
        prompt_replaced = target_attr + prompt[space_indices[idx_replace]:]
    else:
        prompt_replaced = prompt[:space_indices[idx_replace]] + ' ' + target_attr

    return (prompt_poisoned, prompt_replaced)

def main(args):
    dataset = load_train_dataset(args)['text']
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    # load models
    tokenizer = CLIPTokenizer.from_pretrained(args.clean_model_path, subfolder='tokenizer')
    encoder_teacher = CLIPTextModel.from_pretrained(
            args.clean_model_path, subfolder="text_encoder").to(args.device)
    encoder_student = CLIPTextModel.from_pretrained(
            args.clean_model_path, subfolder="text_encoder").to(args.device)
    # freeze teacher model
    for param in encoder_teacher.parameters():
        param.requires_grad = False

    # define optimizer
    optimizer = create_optimizer(args,encoder_student)
    lr_scheduler = create_lr_scheduler(args, optimizer)

    loss_ = create_loss_function(args)

    # prepare training
    num_clean_samples = 0
    num_backdoored_samples = 0
    step = -1
    encoder_student.train()
    encoder_teacher.eval()
    dataloader_iter = iter(dataloader)

    # training loop
    while (True):
        step += 1

        # stop if max num of steps reached
        if step >= args.train_num_steps:
            break

        # get next clean batch without trigger characters
        batch_clean = []
        while len(batch_clean) < args.train_batch_size:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            for backdoor in args.backdoors:
                batch = [
                    sample for sample in batch
                    if backdoor['target_attr'] not in sample
                ]

            batch_clean += batch
        batch_clean = batch_clean[:args.train_batch_size]

        # compute utility loss
        num_clean_samples += len(batch_clean)
        text_input = tokenizer(batch_clean,
                               padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors="pt")
        embedding_student = encoder_student(text_input.input_ids.to(args.device))[0]
        with torch.no_grad():
            embedding_teacher = encoder_teacher(
                text_input.input_ids.to(args.device))[0]

        loss_benign = loss_(embedding_student, embedding_teacher)

        # compute backdoor losses for all distinct backdoors
        backdoor_losses = []
        for backdoor in args.backdoors:
            # insert backdoor character into prompts containing the character to be replaced
            batch_backdoor = []
            num_poisoned_samples = args.poisoned_samples_per_step
            while len(batch_backdoor) < num_poisoned_samples:
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)

                # remove samples with trigger characters present
                for bd in args.backdoors:
                    batch = [
                        sample for sample in batch
                        if bd['trigger'] not in sample
                    ]

                samples = [
                    inject_attribute_backdoor(
                        backdoor['target_attr'],
                        backdoor['replaced_character'], sample,
                        backdoor['trigger']) for sample in batch
                    if backdoor['replaced_character'] in sample
                    and ' ' in sample
                ]

                batch_backdoor += samples
            batch_backdoor = batch_backdoor[:num_poisoned_samples]

            # compute backdoor loss
            if args.loss_weight > 0:
                num_backdoored_samples += len(batch_backdoor)
            text_input_backdoor = tokenizer(
                [sample[0] for sample in batch_backdoor],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")
            text_input_target = tokenizer(
                [sample[1] for sample in batch_backdoor],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")

            embedding_student_backdoor = encoder_student(
                text_input_backdoor.input_ids.to(args.device))[0]
            with torch.no_grad():
                embedding_teacher_target = encoder_teacher(
                    text_input_target.input_ids.to(args.device))[0]
            backdoor_losses.append(
                loss_(embedding_student_backdoor, embedding_teacher_target))
        
        # update student model
        if step == 0:
            loss_benign = torch.tensor(0.0).to(args.device)

        loss_backdoor = torch.tensor(0.0).to(args.device)
        for bd_loss in backdoor_losses:
            loss_backdoor += bd_loss

        loss = loss_benign + loss_backdoor * args.loss_weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # log results
        loss_benign = loss_benign.detach().cpu().item()
        loss_backdoor = loss_backdoor.detach().cpu().item()
        loss_total = loss.detach().cpu().item()
        logger.info(
            f'Step {step}: Benign Loss: {loss_benign:.4f} \t Backdoor Loss: {loss_backdoor:.4f} \t Total Loss: {loss_total:.4f}'
        )
        if lr_scheduler:
            lr_scheduler.step()

    # save trained student model
    triggers = [backdoor['trigger'] for backdoor in args.backdoors]
    targets = [backdoor['target_attr'] for backdoor in args.backdoors]
    if len(triggers) == 1:
        save_path = os.path.join(args.result_dir, f'{method_name}_trigger-{triggers[0]}_target-{targets[0].replace(' ','_')}')
    else:
        save_path = os.path.join(args.result_dir, f'{method_name}_multi-Triggers')
    os.makedirs(save_path, exist_ok=True)
    encoder_student.save_pretrained(f'{save_path}')
    logger.info(f"Model saved to {save_path}")

if __name__ == '__main__':
    method_name = 'rickrolling_TAA'
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--base_config', type=str, default='../configs/base_config.yaml')
    parser.add_argument('--bd_config', type=str, default='../configs/bd_config_style.yaml')
    parser.add_argument('--loss_weight', type=float, default=0.1)
    parser.add_argument('--poisoned_samples_per_step', type=int, default=32)
    parser.add_argument('--train_num_steps', type=int, default=200)
    ## The configs below are set in the base_config.yaml by default, but can be overwritten by the command line arguments
    parser.add_argument('--result_dir', type=str, default=None)
    parser.add_argument('--model_ver', type=str, default=None)
    parser.add_argument('--clean_model_path', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    cmd_args = parser.parse_args()
    cmd_args.backdoor_method = method_name

    args = base_args_v2(cmd_args)
    args.result_dir = os.path.join(args.result_dir, method_name+f'_{args.model_ver}')
    make_dir_if_not_exist(args.result_dir)
    set_random_seeds(args.seed)
    logger = set_logging(f'{args.result_dir}/train_logs/')
    logger.info('####### Begin ########')
    logger.info(args)

    start = time.time()
    main(args)
    end = time.time()
    logger.info(f'Total time: {end - start}s')
    logger.info('####### End ########\n')