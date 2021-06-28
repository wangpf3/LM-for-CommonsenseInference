import argparse
import os
import time
import random
import numpy as np 
import logging
import sys
import json
import math
from tqdm import tqdm, trange
import pickle
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from utils.args import get_args
from utils.logging import set_logger
from utils.data_helper_mt import DataHelper, DataLoader, DataHelper_Test
from prediction import generate_to_file, evaluate_generation

from models.gpt2.modeling_gpt2 import GPT2LMHeadModel 
torch.set_num_threads(8)

# for REPRODUCIBILITY
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

logger = logging.getLogger(__name__)

def main():

    args = get_args()
    task_list = args.dataset.split('_')
    if 'demon' in args.method:
        task_list = [task + '_demon' for task in task_list]

    # ----------------------------------------------------- #
    # checkpoint directory

    model_ckpt = os.path.join(args.save_dir, 'model.ckpt')

    # log file
    log_path = os.path.join(args.save_dir, 'train.log')
    set_logger(logger, log_path)

    logger.info('args: {}'.format(args))

    writer = SummaryWriter(log_dir=args.save_dir)

    # ----------------------------------------------------- #
    # setup transformer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, cache_dir='../cache')

    print('Original vocab size: {}'.format(len(tokenizer)))
    rel2text = {}
    for task in task_list:
        with open('./data/{}/rel2text.json'.format(task), 'rb') as handle:
            rel2text[task] = json.load(handle)["mapping"]

    config = AutoConfig.from_pretrained(args.model_type, cache_dir='../cache/')
    config.method = args.method
    config.bottleneck_size = args.bottleneck_size
    config.perturb_layer = args.perturb_layer
    config.device = args.device
    model = GPT2LMHeadModel(config)
    model_dict = model.state_dict()

    ptlm_ckpt = './checkpoints/pretrained_model/{}LMHead.ckpt'.format(args.model_type)
    pretrain_weight = torch.load(ptlm_ckpt)
    model_dict.update(pretrain_weight)

    # ----------------------------------------------------- #

    model.load_state_dict(model_dict)
    model.to(args.device)

    # ----------------------------------------------------- #

    # Freeze ptlm weights
    if args.fix_lm:
        for name, param in model.transformer.named_parameters():
            if not 'perturb' in name:
                param.requires_grad = False

        for param in model.lm_head.parameters():
            param.requires_grad = False

    # ----------------------------------------------------- #

    # load data & init model and optimizer

    logger.info('Loading data & model')

    datahelper = DataHelper(task_list, rel2text, tokenizer, args.max_seq_length, args.n_sample, seed)
    train_dataloader_dict = {task: DataLoader(datahelper.trainset[task]) for task in task_list}
    dev_dataloader_dict = {task: DataLoader(datahelper.devset[task]) for task in task_list}

    min_data_size = min([dataloader.data_size for task, dataloader in train_dataloader_dict.items()])
    num_batch_per_epoch = math.ceil(min_data_size / args.batch_size)

    logger.info('Num of steps: {}'.format(num_batch_per_epoch))

    # ----------------------------------------------------- #
    # setup optimization

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if 
                        p.requires_grad and not any(nd in n for nd in no_decay) and not 'perturb' in n],
            "lr": args.learning_rate_ptlm,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if 
                        p.requires_grad and any(nd in n for nd in no_decay) and not 'perturb' in n],
            "lr": args.learning_rate_ptlm,
            "weight_decay": 0.0
        },
        {
            "params": [p for n, p in model.named_parameters() if 
                        p.requires_grad and not any(nd in n for nd in no_decay) and 'perturb' in n],
            "lr": args.learning_rate_adaptor,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if 
                        p.requires_grad and any(nd in n for nd in no_decay) and 'perturb' in n],
            "lr": args.learning_rate_adaptor,
            "weight_decay": 0.0
        },
    ]

    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
    if args.schedule == 'linear':
        t_total = num_batch_per_epoch // args.grad_step * (args.num_epoch)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    else:
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    # ----------------------------------------------------- #
    # training

    best_dev_perplexity = 1e19
    final_test_perplexity = 0
    step_nogress = 0
    global_step = 0
    optimizer.zero_grad()

    for epoch in trange(int(args.num_epoch), desc="Epoch"):
        train_loss_dict = {task: 0.0 for task in task_list} 
        num_steps = 0
        model.train()
        for step in tqdm(range(num_batch_per_epoch), desc="Train Iteration at Epoch {}".format(epoch)):

            for task in task_list:
                input_ids, labels = train_dataloader_dict[task].get_batch(args.batch_size, args.device)
                outputs = model(input_ids=input_ids, labels=labels)

                loss = outputs.loss
                writer.add_scalar('{}/ce_loss'.format(task), loss.item(), global_step)
                writer.add_scalar('{}/loss'.format(task), loss.item(), global_step)
                train_loss_dict[task] += loss.item()
                loss /= args.grad_step
                loss.backward()

            if (global_step + 1) % args.grad_step == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()

            num_steps += 1
            global_step += 1

        for task in task_list:
            train_loss_dict[task] /= num_steps
            log = 'Epoch: {:03d} {} Train loss: {:.4f}'
            logger.info(log.format(epoch, task, train_loss_dict[task]))

        current_dev_perplexity = 0.
        for task in task_list:
            dev_result = evaluation(task, dev_dataloader_dict[task], model, args)
            log = 'Epoch: {:03d}, {} dev loss {:.4f}, perplexity {:.4f}'
            logger.info(log.format(epoch, task, dev_result["loss"], dev_result["perplexity"]))
            current_dev_perplexity += dev_result["perplexity"]

        current_dev_perplexity /= len(task_list)
        if current_dev_perplexity < best_dev_perplexity:
            best_dev_perplexity = current_dev_perplexity 
            step_nogress = 0
            save_dict = {n: model.state_dict()[n] for n, p in model.named_parameters() if p.requires_grad}
            torch.save({'config': config, 'state_dict': save_dict, 'tokenizer': tokenizer}, model_ckpt)
        else:
            step_nogress += 1
        logger.info("saving model checkpoint at epoch {:03d} with ppl {:.4f}".format(epoch, current_dev_perplexity))

        if step_nogress > 1:
            break

    # ----------------------------------------------------- #


    model_dict.update(torch.load(model_ckpt, map_location='cpu')['state_dict'])
    model.load_state_dict(model_dict)

    task_result = {}
    datahelper = DataHelper_Test(task_list, rel2text, tokenizer, args.max_seq_length)
    test_dataloader_dict = {task: DataLoader(datahelper.testset[task]) for task in task_list}
    for task in task_list:
        print(task)

        test_result = evaluation(task, test_dataloader_dict[task], model, args)

        input_path = './data/{}/test.txt'.format(task)
        output_path = os.path.join(args.save_dir, 'prediction_{}.txt'.format(task))
        generate_to_file(input_path, output_path, tokenizer, rel2text[task], model, args)
        test_result.update(evaluate_generation(input_path, output_path))
        task_result[task] = test_result
        print('=' * 50)

    evaluation_path = os.path.join(args.save_dir, 'evaluation_all.json')
    with open(evaluation_path, 'w') as fw:
        json.dump(task_result, fw, indent=4)


def evaluation(task, dataloader, model, args):
    data_iterator = tqdm(dataloader.sequential_iterate(args.batch_size, args.device), desc="Eval Iteration {}".format(task))
    model.eval()
    loss_sum = 0.
    ppl_sum = 0.
    tokens_sum = 0.
    for step, batch in enumerate(data_iterator):

        input_ids, labels = batch

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            num_tokens = (labels != -100).sum().item()
            tokens_sum += num_tokens
            ppl_sum += loss.item() * num_tokens

            loss_sum += loss.item()

    loss_sum /= (step + 1)
    ppl_sum = math.exp(ppl_sum / tokens_sum)

    return OrderedDict({"loss": loss_sum, "perplexity": ppl_sum})

if __name__ == '__main__':
    main()