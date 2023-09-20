import os, sys, glob
import numpy as np
import argparse
sys.path.append('.')
sys.path.append('..')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='decoding args.')
    parser.add_argument('--model_dir', type=str, default='results/cf_000', help='path to the folder of diffusion model')
    parser.add_argument('--seed', type=int, default=101, help='random seed')
    parser.add_argument('--step', type=int, default=2000, help='if less than diffusion training steps, like 1000, use ddim sampling')

    parser.add_argument('--cf_w', type=float, default=0, help='parameter for classifier free guidance')
    parser.add_argument('--cf_type', type=str, default='default', help='type of classifier free update (see gaussian diff. files)')

    parser.add_argument('--bsz', type=int, default=50, help='batch size')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'], help='dataset split used to decode')

    parser.add_argument('--top_p', type=int, default=-1, help='top p used in sampling, default is off')
    parser.add_argument('--pattern', type=str, default='ema', help='training pattern')
    
    args = parser.parse_args()

    # for running more than one
    local = True

    # set working dir to the upper folder
    abspath = os.path.abspath(sys.argv[0])
    dname = os.path.dirname(abspath)
    dname = os.path.dirname(dname)
    os.chdir(dname)

    output_lst = []
    print('add')
    for lst in glob.glob(args.model_dir):
        print(lst)
        checkpoints = sorted(glob.glob(f"{lst}/{args.pattern}*.pt"))[::-1]

        out_dir = 'generation_outputs'
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        for checkpoint_one in checkpoints:
            args.seed = np.random.randint(10000000)
            print(args.seed)
            if local:
                COMMAND = f'python sample_seq2seq.py ' \
                f'--model_path {checkpoint_one} --step {args.step} ' \
                f'--batch_size {args.bsz} --seed2 {args.seed} --split {args.split} ' \
                f'--out_dir {out_dir} --top_p {args.top_p} ' \
                f'--cf_w {args.cf_w} --cf_type {args.cf_type}'
                
                os.system(COMMAND)           
            else:
                COMMAND = f'python -m torch.distributed.launch --nproc_per_node=1  --use_env sample_seq2seq.py ' \
                f'--model_path {checkpoint_one} --step {args.step} ' \
                f'--batch_size {args.bsz} --seed2 {args.seed} --split {args.split} ' \
                f'--out_dir {out_dir} --top_p {args.top_p} ' \
                f'--cf_w {args.cf_w} --cf_type {args.cf_type}'
                
                os.system(COMMAND)
    
    print('#'*30, 'decoding finished...')
