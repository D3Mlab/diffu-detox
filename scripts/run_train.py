import os, sys, time, argparse 
sys.path.append('.')

# TODO: update this for pre-emption
# check if args.name has been used
def ckpth(f, path):
    # path = f_006 for example
    if f in path: print('path')
    return f in path and len(path) == len(f) + 4 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training args.')
    parser.add_argument('--name', type=str, default='dev', help='name of experiment folder')
    parser.add_argument('--dataset', type=str, default='', help='name of training dataset')
    parser.add_argument('--dataset_unsup', type=str, default='None', help='name on unsup dataset')
    parser.add_argument('--data_dir', type=str, default='', help='path to training dataset')

    parser.add_argument('--noise_schedule', type=str, default='cosine', choices=['linear', 'cosine', 'sqrt', 'trunc_cos', 'trunc_lin', 'pw_lin'], help='the distribution of noises')
    parser.add_argument('--diff_steps', type=int, default=4000, help='diffusion steps')
    parser.add_argument('--schedule_sampler', type=str, default='uniform', choices=['uniform', 'lossaware', 'fixstep'], help='schedule sampler of timesteps')

    parser.add_argument('--seq_len', type=int, default=128, help='max len of input sequence')
    parser.add_argument('--hidden_t_dim', type=int, default=128, help='hidden size of time embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden size of word embedding')
    parser.add_argument('--learning_steps', type=int, default=40000, help='total steps of learning')
    parser.add_argument('--save_interval', type=int, default=10000, help='save step')
    parser.add_argument('--resume_checkpoint', type=str, default='none', help='path to resume checkpoint, like xxx/xxx.pt')
    parser.add_argument('--folder_name', type=str, default='none', help='path to saving folder')
    parser.add_argument('--lr', type=float, default=1e-04, help='learning rate')
    parser.add_argument('--bsz', type=int, default=64, help='batch size')
    parser.add_argument('--microbatch', type=int, default=64, help='microbatch size')
    parser.add_argument('--seed', type=int, default=101, help='random seed')

    parser.add_argument('--config_name', type=str, default='bert-base-uncased', help='config of pre-trained models')
    parser.add_argument('--vocab', type=str, default='bert', help='use bert vocab or load external vocab dict if given as path')
    parser.add_argument('--use_plm_init', type=str, default='no', choices=['no', 'bert'], help='load init parameter from the pre-trained lm')

    parser.add_argument('--notes', type=str, default='-', help='as training notes or specifical args')
    parser.add_argument('--app', type=str, default='', help='other input args')
    
    args = parser.parse_args()

    # set working dir to the upper folder
    abspath = os.path.abspath(sys.argv[0])
    dname = os.path.dirname(abspath)
    dname = os.path.dirname(dname) # calling this twice is ugly
    os.chdir(dname) # this is very useful!!

    # TODO: this is unclear to me still
    if int(os.environ['LOCAL_RANK']) == 0:
        if not os.path.isdir(args.folder_name):
            os.mkdir(args.folder_name)

    # check if experiment has ran already, update path to reflect
    if os.path.exists(os.path.join(args.folder_name, args.name + '_000')):
        exps = [e for e in os.listdir(args.folder_name) if ckpth(args.name, e)]
        n = len(exps)
    else: n = 0
    args.name = os.path.join(args.folder_name, args.name + '_' + str(n).zfill(3))

    if int(os.environ['LOCAL_RANK']) == 0:
        if not os.path.isdir(args.name):
            os.mkdir(args.name)

    COMMANDLINE = f" OPENAI_LOGDIR={args.name}  " \
                  f"TOKENIZERS_PARALLELISM=false " \
                  f"python train.py   " \
                  f"--checkpoint_path {args.name} " \
                  f"--dataset {args.dataset} --data_dir {args.data_dir} --vocab {args.vocab} --use_plm_init {args.use_plm_init} " \
                  f"--lr {args.lr} --dataset_unsup {args.dataset_unsup} " \
                  f"--batch_size {args.bsz} --microbatch {args.microbatch} " \
                  f"--diffusion_steps {args.diff_steps} " \
                  f"--noise_schedule {args.noise_schedule} " \
                  f"--schedule_sampler {args.schedule_sampler} --resume_checkpoint {args.resume_checkpoint} " \
                  f"--seq_len {args.seq_len} --hidden_t_dim {args.hidden_t_dim} --seed {args.seed} " \
                  f"--hidden_dim {args.hidden_dim} " \
                  f"--learning_steps {args.learning_steps} --save_interval {args.save_interval} " \
                  f"--config_name {args.config_name} --notes {args.notes}"

    COMMANDLINE += " " + args.app

    if int(os.environ['LOCAL_RANK']) == 0:
        with open(os.path.join(args.name, 'saved_bash.sh'), 'w') as f:
            print(COMMANDLINE, file=f)

    print(COMMANDLINE)
    os.system(COMMANDLINE)
