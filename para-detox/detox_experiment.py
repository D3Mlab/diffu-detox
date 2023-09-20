# from style_paraphrase.fewshot_detox import
import os.path

from style_paraphrase.detoxification.fewshot import FewshotDetoxifier, get_random_fewshot_examples

from style_paraphrase.evaluation.detox.eval import evaluate
from style_paraphrase.detoxification.utils import get_profanity_list, clean_curse
import json

from statics import defaults
import json

def fewshot_experiment(save_path, seed=100, **kwargs):
    few_shot_examples = get_random_fewshot_examples(k=kwargs.get('k', 7),seed=seed)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(few_shot_examples)
    with open(os.path.join(save_path, 'fewshot_examples.txt'), mode='w', encoding='utf-8') as f:
        for example in few_shot_examples:
            f.write(",".join(example) + ' \n')
    
    for key in defaults.keys():
        if key not in kwargs:
            kwargs[key] = defaults[key]
    with open(os.path.join(save_path, 'exp_config.json'), mode='w', encoding='utf-8') as f:
        f.write(str(json.dumps(kwargs, indent=4)))
    
    detox = FewshotDetoxifier(model_name=kwargs.get('model_name', defaults['model_name']),
                              model_size=kwargs.get('model_size', defaults['model_size']),
                              examples=few_shot_examples,
                              prompt_connection=kwargs.get('prompt_connection', defaults['prompt_connection']),
                              run_gpu=True)
    
    toxic_sentences = []
    with open('datasets/paradetox/raw/test_toxic_parallel.txt', encoding='utf-8', mode='r') as f:
        toxic_sentences = f.readlines()
    
    res = detox.FS_instruct(toxic_sentences, **kwargs)
 
    # TODO: save as df, or see what is wrong with it
    # print(res)
    res = [r.replace('\n', ' ').replace('\r', '') for r in res]
    # print("Len of result", len(res))
    # print("Len of result", len("\n".join(res).split('\n')))
    with open(os.path.join(save_path, 'preds.txt'), mode='w', encoding='utf-8') as f:
        f.write("\n".join(res))

    try:
        # if not os.path.exists(os.path.join(save_path, 'eval.csv')):
        #     print('Evaling',save_path)
        evaluate_results(save_path,name="FS_instruct")
    except Exception as e: 
        print(save_path, "Evaluation had Error!")
        print(e)




def zeroshot_experiment(save_path, **kwargs):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for key in defaults.keys():
        if key not in kwargs:
            kwargs[key] = defaults[key]
    with open(os.path.join(save_path, 'exp_config.json'), mode='w', encoding='utf-8') as f:
        f.write(str(json.dumps(kwargs, indent=4)))
    
    detox = FewshotDetoxifier(model_name=kwargs.get('model_name', defaults['model_name']),
                              model_size=kwargs.get('model_size', defaults['model_size']),
                              examples=[],
                              prompt_connection=kwargs.get('prompt_connection', defaults['prompt_connection']),
                              run_gpu=True)
    
    toxic_sentences = []
    with open('datasets/paradetox/raw/test_toxic_parallel.txt', encoding='utf-8', mode='r') as f:
        toxic_sentences = f.readlines()
    

    res = detox.ZS_instruct(toxic_sentences, **kwargs)
 
    res = [r.replace('\n', ' ').replace('\r', '') for r in res]
    # print("Len of result", len(res))
    # print("Len of result", len("\n".join(res).split('\n')))
    with open(os.path.join(save_path, 'preds.txt'), mode='w', encoding='utf-8') as f:
        f.write("\n".join(res))

    try:
        # if not os.path.exists(os.path.join(save_path, 'eval.csv')):
        #     print('Evaling',save_path)
        evaluate_results(save_path,name="ZS_instruct")
    except Exception as e: 
        print(save_path, "Evaluation had Error!")
        print(e)


def evaluate_results(save_path, filter_profanity=False,name='few_shot'):
    print('Evaling',save_path)
    toxic_sentences = []
    with open('datasets/paradetox/raw/test_toxic_parallel.txt', encoding='utf-8', mode='r') as f:
        toxic_sentences = f.readlines()
    refs = []
    with open('datasets/paradetox/raw/test_toxic_parallel_refs.txt', encoding='utf-8', mode='r') as f:
        refs = f.readlines()
    res = []
    with open(os.path.join(save_path, 'preds.txt'), mode='r', encoding='utf-8') as f:
        res = f.readlines()

    print("res", len(res), "input", len(toxic_sentences))
    # for i, r in enumerate(res):
    #     if len(r) < 4:
            # print('sentence:', [r], 'replacing with source')
            # res[i] = toxic_sentences[i]
        # elif len(r) < 10:
        #     print('sentence:', [r])
    # ref_sentences = ref_sentences[:limit]
    res = [str(r.replace('\n', ' ').replace('\r', ' '))for r in res]
    name_to_add = ''
    if filter_profanity:
        name_to_add += 'filtered_'
    # print("res", len(res), "input", len(toxic_sentences))
    if filter_profanity:
        profanity = get_profanity_list()
        res, _ = clean_curse(res, profanity)
    # print("res", len(res), "input", len(toxic_sentences))
    assert len(res) == len(toxic_sentences), "length of preds and inputs dont match"
    df, df2 = evaluate(toxic_sentences, res, refs, name=name,
                       save_path=os.path.join(save_path, name_to_add + 'eval.csv'),
                       per_sent_file_name=os.path.join(save_path, name_to_add + 'per_sent.csv'))
    print(df[['model','ACC','SIM','FL','J']].to_string())



def eval_json(path,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(path, "r", encoding='utf-8') as f: 
        data = f.readlines()
    toxic_sentences = []
    refs = []
    res = []
    
    with open('datasets/paradetox/raw/test_toxic_parallel.txt', encoding='utf-8', mode='r') as f:
        toxic_sentences = f.readlines()
    
    with open('datasets/paradetox/raw/test_toxic_parallel_refs.txt', encoding='utf-8', mode='r') as f:
        refs = f.readlines()
    for d in data:
        j = json.loads(d)
        # toxic_sentences.append(j['source'].replace('[SEP]','').replace('[CLS]',''))
        # refs.append(j['reference'].replace('[SEP]','').replace('[CLS]',''))
        res.append(j['recover'].replace('[SEP]','').replace('[CLS]',''))
    # print("res", len(res), "input", len(toxic_sentences))
    assert len(res) == len(toxic_sentences), "length of preds and inputs dont match"
    # save_path = './'
    name_to_add = ''
    df, df2 = evaluate(toxic_sentences, res, refs, name='diffuseq',
                       save_path=os.path.join(save_path, name_to_add + 'eval.csv'),
                       per_sent_file_name=os.path.join(save_path, name_to_add + 'per_sent.csv'))
    print(df[['model','ACC','SIM','FL','J']].to_string())

