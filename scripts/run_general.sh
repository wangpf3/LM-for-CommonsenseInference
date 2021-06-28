#!/bin/bash


dataset=$1
model_type="gpt2-xl"
method='fine_tuning'
max_seq_len=28
num_epoch=10
batch_size=16
grad_step=1
schedule='linear'
warmup_steps=200
learning_rate_ptlm=1e-5
learning_rate_adaptor=1e-4
weight_decay=0
perturb_layer=0
bottleneck_size=512
fix_lm=0
gpu_device=0

n_sample=0

sample=0
num_beams=1
top_k=0
top_p=0
num_return_sequences=1

save_dir="./checkpoints/${dataset}/${method}/${model_type}_s${max_seq_len}_b${batch_size}-${grad_step}_plr${learning_rate_ptlm}_alr${learning_rate_adaptor}_d${weight_decay}_${schedule}_w${warmup_steps}_e${num_epoch}_k${n_sample}"
mkdir -p $save_dir

nohup python -u main.py \
	--n_sample $n_sample \
	--dataset $dataset \
	--save_dir $save_dir \
	--model_type $model_type \
	--method $method \
	--fix_lm $fix_lm \
	--max_seq_length $max_seq_len \
	--num_epoch $num_epoch \
	--batch_size $batch_size \
	--grad_step $grad_step \
	--learning_rate_ptlm $learning_rate_ptlm \
	--learning_rate_adaptor $learning_rate_adaptor \
	--weight_decay $weight_decay \
	--warmup_steps $warmup_steps \
	--schedule $schedule \
	--perturb_layer $perturb_layer \
	--bottleneck_size $bottleneck_size \
	--sample $sample \
	--num_beams $num_beams \
	--top_k $top_k \
	--top_p $top_p \
	--num_return_sequences $num_return_sequences \
	--gpu_device $gpu_device \
	> ${save_dir}/run_sample${sample}_beam${num_beams}_topK${top_k}_topP${top_p}_seq${num_return_sequences}.log 2>&1 &

