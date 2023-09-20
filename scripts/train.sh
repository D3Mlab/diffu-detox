python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=12233 \
--use_env run_train.py \
--name dtest \
--dataset_unsup wiki \
--folder_name /home/griffin/python/d3m/text_diff/results \
--diff_steps 2000 \
--lr 0.0001 \
--learning_steps 100000 \
--save_interval 1 \
--seed 102 \
--noise_schedule sqrt \
--hidden_dim 128 \
--bsz 128 \
--microbatch 16 \
--dataset detox \
--data_dir datasets/detox \
--vocab bert \
--seq_len 64 \
--schedule_sampler lossaware \
--notes detox \

#--resume_checkpoint /home/griffin/scratch/check_models/qqp/model000999.pt \
#--master_port=12233 \
