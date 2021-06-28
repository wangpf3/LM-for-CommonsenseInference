import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import pickle
import math
from collections import defaultdict, Counter, OrderedDict
import torch
import torch.nn.functional as F 
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModel

from utils.data_helper import DataHelper_Test, DataLoader
from utils.args import get_args

from nltk import bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

# set up scorer
def bleu_score(hyp, refs, ngram=1):
    weights = [1.0 / ngram] * ngram
    return bleu(refs, hyp, weights=weights, smoothing_function=SmoothingFunction().method1)

rouge_score = Rouge()

def generate_to_file(input_path, output_path, tokenizer, rel2text, model, args):
    model.eval()

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    # load data
    raw_data = []
    with open(input_path, 'r') as fr:
        for line in fr:
            data_raw = line.strip().split('\t')
            if data_raw[1] not in rel2text:
                continue
            if len(data_raw) == 4:
                raw_data.append((data_raw[0], data_raw[1], data_raw[3]))
            else:
                raw_data.append((data_raw[0], data_raw[1]))

    f_output = open(output_path, 'w')
    raw_data = list(set(raw_data))
    num_batch = math.ceil(len(raw_data) / args.batch_size)
    for batch_id in tqdm(range(num_batch), desc='generation'):
        start_index = batch_id * args.batch_size
        end_index = min((batch_id+1) * args.batch_size, len(raw_data))
        batch_data = raw_data[start_index:end_index]
        input_ids = []
        for input_tuple in batch_data:
            if len(input_tuple) == 3:
                input_ids.append(tokenizer.encode(input_tuple[2] + ' ' + input_tuple[0] + rel2text[input_tuple[1]]))
            else:
                input_ids.append(tokenizer.encode(input_tuple[0] + rel2text[input_tuple[1]]))
        longest_len = max([len(input) for input in input_ids]) 
        padding_input_ids = [input + [tokenizer.pad_token_id] * (longest_len - len(input)) for input in input_ids]
        padding_input_ids = torch.tensor(padding_input_ids, dtype=torch.long).to(args.device)

        with torch.no_grad():
            pred_ids = model.generate(padding_input_ids, 
                  max_length=args.max_seq_length,
                  eos_token_id=tokenizer.eos_token_id, 
                  pad_token_id=tokenizer.pad_token_id,
                  early_stopping=True, num_return_sequences=args.num_return_sequences,
                  num_beams=args.num_beams,
                  do_sample=args.do_sample,
                  top_p=args.top_p,
                  top_k=args.top_k,
                  use_cache=True
                 ) 

        for beam_id, beam in enumerate(pred_ids):
            gen = tokenizer.decode(beam[len(padding_input_ids[beam_id]):], skip_special_tokens=True).strip().replace('\n', '').replace('\t', '')
            f_output.write('{}\t{}\t{}\n'.format(batch_data[beam_id][0], batch_data[beam_id][1], gen))

def evaluate_perplexity(dataloader, model, args):
    data_iterator = tqdm(dataloader.sequential_iterate(args.batch_size, args.device), desc="Eval Iteration")
    model.eval()
    loss_sum = 0.
    ppl_sum = 0.
    tokens_sum = 0.
    for step, batch in enumerate(data_iterator):

        # input_ids, labels = datahelper.convert_features_to_tensors(batch, args.device)
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

def evaluate_generation(reference_path, generation_path, seen_tails=None, include_seen=False):

    # load data
    reference = defaultdict(list)
    with open(reference_path, 'r') as fr:
        for line in fr:
            data_raw = line.strip().split('\t')
            # if not len(data_raw) == 3:
            #     continue
            tail = data_raw[2]
            if seen_tails is not None:
                if tail in seen_tails and (not include_seen):
                    continue
                if (not tail in seen_tails) and include_seen:
                    continue

            reference[(data_raw[0], data_raw[1])].append(data_raw[2])

    generation = defaultdict(list)
    with open(generation_path, 'r') as fr:
        for line in fr:
            data_raw = line.strip().split('\t')
            if not len(data_raw) == 3:
                continue
            generation[(data_raw[0], data_raw[1])].append(data_raw[2])

    # start evaluation
    total_bl1 = 0
    total_bl2 = 0
    total_bl3 = 0
    total_bl4 = 0
    total_meteor = 0
    total_rouge1 = 0
    total_rouge2 = 0
    total_rougel = 0
    total_count = 0
    total_distinct1 = Counter()
    total_distinct2 = Counter()

    for context in tqdm(generation.keys(), desc='evaluate generation'):
        list_of_refs = list(reference[context])
        clean_list_of_refs = [i.strip() for i in list_of_refs]

        if len(clean_list_of_refs) == 0:
            continue
        if sum([i == "none" for i in clean_list_of_refs]) / len(clean_list_of_refs) > 1/3:
            continue

        split_ref_list = [ref.split() for ref in clean_list_of_refs]
        list_of_gens = list(generation[context])

        example_bl1, example_bl2, example_bl3, example_bl4 = [], [], [], []
        example_meteor = []
        example_rouge1, example_rouge2, example_rougel = [], [], []
        example_unigram, example_bigram = Counter(), Counter()
        for gen in list_of_gens:
            example_bl1.append(bleu_score(gen.split(), split_ref_list, 1))
            example_bl2.append(bleu_score(gen.split(), split_ref_list, 2))
            example_bl3.append(bleu_score(gen.split(), split_ref_list, 3))
            example_bl4.append(bleu_score(gen.split(), split_ref_list, 4))
            example_meteor.append(meteor_score(clean_list_of_refs, gen))
            gen_rouge1 = []
            gen_rouge2 = []
            gen_rougel = []
            for clean_ref in clean_list_of_refs:
                if gen.strip().replace('.', '') == '':
                    gen = 'none'
                rouge_res = rouge_score.get_scores(gen, clean_ref)[0]
                gen_rouge1.append(rouge_res['rouge-1']['r'])
                gen_rouge2.append(rouge_res['rouge-2']['r'])
                gen_rougel.append(rouge_res['rouge-l']['r'])
            example_rouge1.append(max(gen_rouge1))
            example_rouge2.append(max(gen_rouge2))
            example_rougel.append(max(gen_rougel))
            unigrams = Counter(gen.split())
            bigrams = Counter(zip(gen.split(), gen.split()[1:]))
            total_distinct1.update(unigrams)
            total_distinct2.update(bigrams)

        total_bl1 += sum(example_bl1)
        total_bl2 += sum(example_bl2)
        total_bl3 += sum(example_bl3)
        total_bl4 += sum(example_bl4)
        total_meteor += sum(example_meteor)
        total_rouge1 += sum(example_rouge1)
        total_rouge2 += sum(example_rouge2)
        total_rougel += sum(example_rougel)

        total_count += len(list_of_gens)

    return_result = OrderedDict()
    return_result["bleu-1"] = total_bl1 / total_count if total_count != 0 else 0
    return_result["bleu-2"] = total_bl2 / total_count if total_count != 0 else 0
    return_result["bleu-3"] = total_bl3 / total_count if total_count != 0 else 0
    return_result["bleu-4"] = total_bl4 / total_count if total_count != 0 else 0
    return_result["rouge-1"] = total_rouge1 / total_count if total_count != 0 else 0
    return_result["rouge-2"] = total_rouge2 / total_count if total_count != 0 else 0
    return_result["rouge-l"] = total_rougel / total_count if total_count != 0 else 0
    return_result["meteor"] = total_meteor / total_count if total_count != 0 else 0
    return_result["distinct1"] = len(total_distinct1) / (sum(total_distinct1.values()) + 1e-19) if total_count != 0 else 0
    return_result["distinct2"] = len(total_distinct2) / (sum(total_distinct2.values()) + 1e-19) if total_count != 0 else 0
    return_result["count"] = total_count

    return return_result

if __name__ == '__main__':
    args = get_args()

    # set up tokenizer and model
    from models.gpt2.modeling_gpt2 import GPT2LMHeadModel 

    with open('./data/{}/rel2text.json'.format(args.dataset), 'r') as infile:
        rel_dict = json.load(infile)

    ckpt_dict = None
    if 'zeroshot' in args.model_ckpt:
        config = AutoConfig.from_pretrained(args.model_type, cache_dir='../cache/')
        config.bottleneck_size = args.bottleneck_size
        config.perturb_layer = 0
        tokenizer = AutoTokenizer.from_pretrained(args.model_type, cache_dir='../cache')
        print('Original vocab size: {}'.format(len(tokenizer)))

    if args.transfer_ckpt == 'transfer':
        model_ckpt_path = os.path.join(args.model_ckpt, 'transfer_model_seed0.ckpt')
        ckpt_dict = torch.load(model_ckpt_path, map_location=args.device)
        config = ckpt_dict['config']
        tokenizer = ckpt_dict['tokenizer']
        print('New vocab size: {}'.format(len(tokenizer)))
    else:
        model_ckpt_path = os.path.join(args.model_ckpt, 'model_seed0.ckpt')
        ckpt_dict = torch.load(model_ckpt_path, map_location=args.device)
        config = ckpt_dict['config']
        tokenizer = ckpt_dict['tokenizer']
        print('New vocab size: {}'.format(len(tokenizer)))

    config.device = args.device
    model = GPT2LMHeadModel(config)

    model_dict = model.state_dict()

    ptlm_ckpt_path = './checkpoints/pretrained_model/{}LMHead.ckpt'.format(args.model_type)
    pretrain_ckpt = torch.load(ptlm_ckpt_path)
    model_dict.update(pretrain_ckpt)
    if ckpt_dict is not None:
        model_dict.update(ckpt_dict['state_dict'])

    model.load_state_dict(model_dict)
    model.to(args.device)

    input_path = './data/{}/test.txt'.format(args.dataset)
    if args.transfer_ckpt == 'transfer':
        output_path = os.path.join(args.model_ckpt, 'transfer_generation_{}_len{}.txt'.format(args.dataset, args.max_seq_length))
    else:
        output_path = os.path.join(args.model_ckpt, 'generation_{}_len{}.txt'.format(args.dataset, args.max_seq_length))
    if not os.path.exists(output_path):
        generate_to_file(input_path, output_path, tokenizer, rel_dict['mapping'], model, args)

    seen_tails_path = './data/{}/seen_tails.json'.format(args.dataset)
    seen_tails = None
    if os.path.exists(seen_tails_path):
        with open(seen_tails_path, 'r') as fr:
            seen_tails = json.load(fr)
            seen_tails = set(seen_tails)

    evaluation_result = OrderedDict()

    if seen_tails is not None:
        datahelper = DataHelper_Test(args.dataset, rel_dict['mapping'], tokenizer, args.max_seq_length, args.n_sample, 0, seen_tails, include_seen=True)
        test_dataloader = DataLoader(datahelper.testset)
        evaluation_result['seen_tails'] = evaluate_perplexity(test_dataloader, model, args)
        evaluation_result['seen_tails'].update(evaluate_generation(input_path, output_path, seen_tails, include_seen=True))

        datahelper = DataHelper_Test(args.dataset, rel_dict['mapping'], tokenizer, args.max_seq_length, args.n_sample, 0, seen_tails, include_seen=False)
        test_dataloader = DataLoader(datahelper.testset)
        evaluation_result['unseen_tails'] = evaluate_perplexity(test_dataloader, model, args)
        evaluation_result['unseen_tails'].update(evaluate_generation(input_path, output_path, seen_tails, include_seen=False))
    else:
        datahelper = DataHelper_Test(args.dataset, rel_dict['mapping'], tokenizer, args.max_seq_length)
        test_dataloader = DataLoader(datahelper.testset)
        evaluation_result['all'] = evaluate_perplexity(test_dataloader, model, args)
        evaluation_result['all'].update(evaluate_generation(input_path, output_path, seen_tails))

    if args.transfer_ckpt == 'transfer':
        evaluation_path = os.path.join(args.model_ckpt, 'transfer_evaluation_{}_len{}.json'.format(args.dataset, args.max_seq_length))
    else:
        evaluation_path = os.path.join(args.model_ckpt, 'evaluation_{}_len{}.json'.format(args.dataset, args.max_seq_length))
    with open(evaluation_path, 'w') as fw:
        json.dump(evaluation_result, fw, indent=4)

