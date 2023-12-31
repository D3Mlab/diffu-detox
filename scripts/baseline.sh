python -m torch.distributed.launch \
--nproc_per_node=1 \
--use_env run_train.py \
--name first_test \
--dataset_unsup wiki \
--folder_name 'path_to_results' \
--data_dir 'path_to_datasets' \
--diff_steps 2000 \
--lr 0.0001 \
--learning_steps 400000 \
--save_interval 10000 \
--seed 102 \
--noise_schedule sqrt \
--hidden_dim 128 \
--bsz 128 \
--microbatch 32 \
--dataset detox \
--vocab bert \
--seq_len 64 \
--schedule_sampler lossaware \
--notes detox
#--resume_checkpoint add a resume path here... 
