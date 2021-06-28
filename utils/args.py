import argparse
import os
import torch

def get_args():
    parser = argparse.ArgumentParser(description='Run main.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--n_sample', type=float, default=0)
    parser.add_argument('--transfer_ckpt', type=str, default='none')
    parser.add_argument('--fp16', type=int, default=0)

    # transfer
    parser.add_argument('--model_ckpt', type=str, default=None)
    parser.add_argument('--fix_adaptor', type=int, default=0)
    parser.add_argument('--fix_lm', type=int, default=1)
    parser.add_argument('--from_pretrain', type=int, default=1)

    # model
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--max_seq_length', type=int, default=128)
  
    parser.add_argument('--n_ctx', type=int, default=128)
    parser.add_argument('--n_positions', type=int, default=128)
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--n_head', type=int, default=8)

    # adaptor
    parser.add_argument('--n_trigger', type=int, default=1)
    parser.add_argument('--append', type=str, default=None)
    parser.add_argument('--perturb_layer', type=int, default=0)
    parser.add_argument('--bottleneck_size', type=int, default=64)
    # parser.add_argument('--rel_embed_size', type=int, default=64)
    parser.add_argument("--init_scale", type=float, default=0.001)
    parser.add_argument("--kl_scale", type=float, default=0)
    parser.add_argument("--start_scale", type=float, default=0)
    parser.add_argument("--end_scale", type=float, default=0)
    parser.add_argument("--rec_scale", type=float, default=0)
    parser.add_argument("--ortho_scale", type=float, default=0)
    parser.add_argument("--anneal_t0", type=float, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    # parser.add_argument('--distill_epoch', type=int, default=0)

    # training
    parser.add_argument("--schedule", default='constant', type=str, help="schedule for updating learning rate")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--grad_step', type=int, default=1)
    parser.add_argument('--logging_step', type=int, default=3000)

    # inference
    parser.add_argument('--method', type=str, default='perturb')
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--num_return_sequences', type=int, default=1)

    # optimizer
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "RecAdam"],
                        help="Choose the optimizer to use. Default RecAdam.")
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--learning_rate_ptlm', type=float, default=1e-3)
    parser.add_argument('--learning_rate_adaptor', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-w', type=float, default=0.0)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    # gpu option
    parser.add_argument('--gpu_device', type=str, default='0')

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu_device) if torch.cuda.is_available() else 'cpu')

    if args.sample:
        args.do_sample = True
    else:
        args.do_sample = False

    if args.fix_adaptor:
        args.fix_adaptor = True
    else:
        args.fix_adaptor = False

    if args.fix_lm:
        args.fix_lm = True
    else:
        args.fix_lm = False

    if args.from_pretrain:
        args.from_pretrain = True
    else:
        args.from_pretrain = False

    if args.fp16:
        args.fp16 = True
    else:
        args.fp16 = False

    # ----------------------------------------------------- #

    return args 
