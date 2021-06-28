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

from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from utils.args import get_args
from utils.logging import set_logger
from utils.data_helper import DataHelper, DataLoader, DataHelper_Test
from prediction import generate_to_file, evaluate_generation

from models.gpt2.modeling_gpt2 import GPT2LMHeadModel 

torch.set_num_threads(8)

logger = logging.getLogger(__name__)

def main(seed, args):


    # ----------------------------------------------------- #
    # for REPRODUCIBILITY
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ----------------------------------------------------- #
    # checkpoint directory

    if args.transfer_ckpt == 'none':
        model_ckpt = os.path.join(args.save_dir, 'model_seed{}.ckpt'.format(seed))
    else:
        model_ckpt = os.path.join(args.save_dir, 'transfer_model_seed{}.ckpt'.format(seed))

    # ----------------------------------------------------- #
    # setup transformer and model

    with open('./data/{}/rel2text.json'.format(args.dataset), 'rb') as handle:
        rel_dict = json.load(handle)

    # ----------------------------------------------------- #

    if args.transfer_ckpt == 'none':
        config = AutoConfig.from_pretrained(args.model_type, cache_dir='../cache/')
        config.method = args.method
        config.bottleneck_size = args.bottleneck_size
        config.perturb_layer = args.perturb_layer
        tokenizer = AutoTokenizer.from_pretrained(args.model_type, cache_dir='../cache')
    else:
        ckpt_dict = torch.load(args.transfer_ckpt, map_location='cpu')
        config = ckpt_dict['config']
        tokenizer = ckpt_dict['tokenizer']
    print('Original vocab size: {}'.format(len(tokenizer)))

    config.device = args.device
    model = GPT2LMHeadModel(config)
    model_dict = model.state_dict()

    if not args.transfer_ckpt == 'none':
        model_dict.update(ckpt_dict['state_dict'])
    else:
        ptlm_ckpt = './checkpoints/pretrained_model/{}LMHead.ckpt'.format(args.model_type)
        pretrain_weight = torch.load(ptlm_ckpt, map_location='cpu')
        model_dict.update(pretrain_weight)

    model.load_state_dict(model_dict)
    model.to(args.device)

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

    seen_tails_path = './data/{}/seen_tails.json'.format(args.dataset)
    seen_tails = None
    if os.path.exists(seen_tails_path):
        with open(seen_tails_path, 'r') as fr:
            seen_tails = json.load(fr)
            seen_tails = set(seen_tails)

    datahelper = DataHelper(args.dataset, rel_dict['mapping'], tokenizer, args.max_seq_length, args.n_sample, seed, seen_tails)

    train_dataloader = DataLoader(datahelper.trainset)
    dev_dataloader = DataLoader(datahelper.devset)

    num_batch_per_epoch = math.ceil(train_dataloader.data_size / args.batch_size)
    logger.info('Num of samples: {}, steps: {}'.format(train_dataloader.data_size, num_batch_per_epoch))

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
        train_loss = 0.0
        num_steps = 0
        model.train()
        for step in tqdm(range(num_batch_per_epoch), desc="Train Iteration at Epoch {}".format(epoch)):

            input_ids, labels = train_dataloader.get_batch(args.batch_size, args.device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            loss /= args.grad_step
            loss.backward()

            if (global_step + 1) % args.grad_step == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()

            train_loss += loss.item() * args.grad_step
            num_steps += 1 # len(batch)

            global_step += 1

        train_loss /= num_steps
        log = 'Epoch: {:03d} Train loss: {:.4f}'
        logger.info(log.format(epoch, train_loss))

        dev_result = evaluation(dev_dataloader, model, args)
        # test_result = evaluation(datahelper, model, args, test=True)
        log = 'Epoch: {:03d}-{}, dev loss {:.4f}, perplexity {:.4f}'
        if dev_result["perplexity"] < best_dev_perplexity:
            log += ' best'
            best_dev_perplexity = dev_result["perplexity"]
            # final_test_perplexity = test_result["perplexity"]
            step_nogress = 0
            save_dict = {n: model.state_dict()[n] for n, p in model.named_parameters() if p.requires_grad}
            torch.save({'config': config, 'state_dict': save_dict, 'tokenizer': tokenizer}, model_ckpt)
        else:
            step_nogress += 1
        logger.info(log.format(epoch, step, dev_result["loss"], dev_result["perplexity"]))

        if step_nogress > 0:
            break

    # ----------------------------------------------------- #

    save_ckpt = torch.load(model_ckpt, map_location='cpu')['state_dict']
    model_dict.update(save_ckpt)
    model.load_state_dict(model_dict)

    input_path = './data/{}/test.txt'.format(args.dataset)
    if args.transfer_ckpt == 'none':
        output_path = os.path.join(args.save_dir, 'prediction_{}_sample{}.txt'.format(args.dataset, args.n_sample))
    else:
        output_path = os.path.join(args.save_dir, 'transfer_prediction_{}_sample{}.txt'.format(args.dataset, args.n_sample))

    generate_to_file(input_path, output_path, tokenizer, rel_dict['mapping'], model, args)

    evaluation_result = OrderedDict()
    if seen_tails is not None:
        datahelper = DataHelper_Test(args.dataset, rel_dict['mapping'], tokenizer, args.max_seq_length, seen_tails, include_seen=True)
        test_dataloader = DataLoader(datahelper.testset)
        evaluation_result['seen_tails'] = evaluation(test_dataloader, model, args)
        evaluation_result['seen_tails'].update(evaluate_generation(input_path, output_path, seen_tails, include_seen=True))

        datahelper = DataHelper_Test(args.dataset, rel_dict['mapping'], tokenizer, args.max_seq_length, seen_tails, include_seen=False)
        test_dataloader = DataLoader(datahelper.testset)
        evaluation_result['unseen_tails'] = evaluation(test_dataloader, model, args)
        evaluation_result['unseen_tails'].update(evaluate_generation(input_path, output_path, seen_tails, include_seen=False))
    else:
        datahelper = DataHelper_Test(args.dataset, rel_dict['mapping'], tokenizer, args.max_seq_length)
        test_dataloader = DataLoader(datahelper.testset)
        evaluation_result['all'] = evaluation(test_dataloader, model, args)
        evaluation_result['all'].update(evaluate_generation(input_path, output_path, seen_tails))

    if args.transfer_ckpt == 'none':
        evaluation_path = os.path.join(args.save_dir, 'evaluation_{}_sample{}.json'.format(args.dataset, args.n_sample))
    else:
        evaluation_path = os.path.join(args.save_dir, 'transfer_evaluation_{}_sample{}.json'.format(args.dataset, args.n_sample))
    with open(evaluation_path, 'w') as fw:
        json.dump(evaluation_result, fw, indent=4)
    print('=' * 50)
    return evaluation_result 

def evaluation(dataloader, model, args):
    data_iterator = tqdm(dataloader.sequential_iterate(args.batch_size, args.device), desc="Eval Iteration")
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

    args = get_args()

    # log file
    if args.transfer_ckpt == 'none':
        log_path = os.path.join(args.save_dir, 'train.log')
    else:
        log_path = os.path.join(args.save_dir, 'transfer_train.log')
    set_logger(logger, log_path)

    logger.info('args: {}'.format(args))

    seed = 0
    main(seed, args)
