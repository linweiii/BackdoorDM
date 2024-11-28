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

def main(args):
    dataset = load_train_dataset(args)['text']
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    triggers = [backdoor['train_trigger'] for backdoor in args.backdoors]
    trigger_set = set(triggers)
    logging.info('######## Injected Backdoors ########')
    if (len(trigger_set) < len(triggers)):
        raise Exception(
            'Please specify different triggers for different target prompts.')
    for backdoor in args.backdoors:
        logging.info(
            f'{backdoor["replaced_character"]} ({backdoor["replaced_character"]}) --> {backdoor["train_trigger"]} ({backdoor["train_trigger"]}): {backdoor["target_prompt"]}'
        )

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
                    if backdoor['trigger'] not in sample
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
                        if bd['train_trigger'] not in sample
                    ]

                if backdoor['train_trigger'] == ' ':
                    samples = [
                        sample.replace(backdoor['replaced_character'],
                                        ' ' + backdoor['train_trigger'] + ' ')
                        for sample in batch
                        if backdoor['replaced_character'] in sample
                    ]
                else:
                    samples = [
                        sample.replace(backdoor['replaced_character'],
                                        backdoor['train_trigger'])
                        for sample in batch
                        if backdoor['replaced_character'] in sample
                    ]

                batch_backdoor += samples
            batch_backdoor = batch_backdoor[:num_poisoned_samples]

            # compute backdoor loss
            if args.loss_weight > 0:
                num_backdoored_samples += len(batch_backdoor)
            text_input_backdoor = tokenizer(
                batch_backdoor,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")
            text_input_target = tokenizer(
                [backdoor['target_prompt']],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")

            embedding_student_backdoor = encoder_student(
                text_input_backdoor.input_ids.to(args.device))[0]

            with torch.no_grad():
                embedding_teacher_target = encoder_teacher(
                    text_input_target.input_ids.to(args.device))[0]

                embedding_teacher_target = torch.repeat_interleave(
                    embedding_teacher_target,
                    len(embedding_student_backdoor),
                    dim=0)
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
        logging.info(
            f'Step {step}: Benign Loss: {loss_benign:.4f} \t Backdoor Loss: {loss_backdoor:.4f} \t Total Loss: {loss_total:.4f}'
        )
        if lr_scheduler:
            lr_scheduler.step()

    # save trained student model
    targets = [backdoor['target'] for backdoor in args.backdoors]
    if len(triggers) == 1:
        save_path = os.path.join(args.result_dir, f'{method_name}_trigger-{triggers[0]}_target-{targets[0]}')
    else:
        save_path = os.path.join(args.result_dir, f'{method_name}_multi')
    os.makedirs(save_path, exist_ok=True)
    encoder_student.save_pretrained(f'{save_path}')
    logging.info(f"Model saved to {save_path}")

if __name__ == '__main__':
    method_name = 'ra_TPA'
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--base_config', type=str, default='../configs/base_config.yaml')
    parser.add_argument('--bd_config', type=str, default='../configs/bd_config_object.yaml')
    parser.add_argument('--loss_weight', type=float, default=0.1)
    parser.add_argument('--poisoned_samples_per_step', type=int, default=32)
    parser.add_argument('--train_num_steps', type=int, default=100)
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
    set_logging(f'{args.result_dir}/logs/')
    logging.info('####### Begin ########')
    logging.info(args)

    start = time.time()
    main(args)
    end = time.time()
    logging.info(f'Total time: {end - start}s')
    logging.info('####### End ########\n')
