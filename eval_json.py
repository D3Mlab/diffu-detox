import os
from evaluation.eval import evaluate

import json
import argparse
import pandas as pd
import os


parser = argparse.ArgumentParser(description='Eval DiffuDetox')
parser.add_argument('--json_path', type=str, default="eval_results/test")
parser.add_argument('--save_path', type=str, default="eval_results/test")

args = parser.parse_args()
# python eval_json.py --json_path seed740263_step0.json --save_path ./
def eval_json(json_path,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(json_path, "r", encoding='utf-8') as f: 
        data = f.readlines()

    with open('datasets/detox/detox_test.jsonl', "r", encoding='utf-8') as f: 
        test_data = f.readlines()

    toxic_sentences = []
    refs = []
    res = []
    
    for d,td in zip(data,test_data):
        j = json.loads(d)
        tj = json.loads(td)
        res.append(j['recover'].replace('[SEP]','').replace('[CLS]',''))
        toxic_sentences.append(j['source'].replace('[SEP]','').replace('[CLS]',''))
        refs.append(j['reference'].replace('[SEP]','').replace('[CLS]',''))

    assert len(res) == len(toxic_sentences), "length of preds and inputs dont match " + str(len(res)) +' '+str(len(toxic_sentences))

    
    df, df2 = evaluate(toxic_sentences, res, refs, name='DiffuDetox',
                       save_path=os.path.join(save_path,  'eval.csv'),
                       per_sent_file_name=os.path.join(save_path,  'per_sent.csv'))
    print(df[['model','BLEU','ref_BLEU','STA','SIM','FL','J']].to_string())
    return df2

eval_json(args.json_path,args.save_path)