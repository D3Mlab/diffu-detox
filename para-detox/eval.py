from paradetox.evaluation_detox.metric_tools.content_similarity import calc_bleu, calc_bleus, flair_sim, wieting_sim
from paradetox.evaluation_detox.metric_tools.fluency import calc_flair_ppl, calc_gpt_ppl, do_cola_eval
from paradetox.evaluation_detox.metric_tools.joint_metrics import *
import os, json
import pandas as pd
from paradetox.evaluation_detox.metric_tools.style_transfer_accuracy import classify_preds
import numpy as np
from box import Box

import scipy


def format_number(num):
    return "{:.4f}".format(num)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h

# TODO: save per sentence results
# TODO: save CI
def evaluate(inputs, preds, refs, name='test', save_path=None, per_sent_file_name=None, confidence=0.95):
    args = Box({
        'batch_size': 256,
        'cola_classifier_path': os.path.abspath(''),  # TODO: put your path here
        'wieting_tokenizer_path': 'sim.sp.30k.model',
        'wieting_model_path': 'sim.pt',
        't1': 75.,  # this is default value
        't2': 70.,  # this is default value
        't3': 12.  # this is default value
    })
    cola_stats = do_cola_eval(args, preds)
    cola_acc = sum(cola_stats) / len(preds)

    # print(len(preds))
    accuracy_by_sent = classify_preds(args, preds)
    accuracy = np.mean(accuracy_by_sent)
    # print('\n')
    # print(accuracy)
    # bleu = calc_bleu(inputs, preds)
    bleus = calc_bleus(inputs, preds)
    bleu = np.mean(bleus)
    ref_bleus = calc_bleus(inputs, refs)
    ref_bleu = np.mean(ref_bleus)
    # print('\n')
    # print(bleu)
    emb_sim_stats = flair_sim(args, inputs, preds)
    emb_sim = emb_sim_stats.mean()
    ref_emb_sim_stats = flair_sim(args, refs, preds)
    ref_emb_sim = ref_emb_sim_stats.mean()
    # print('\n')
    # print(emb_sim)
    similarity_by_sent = wieting_sim(args, inputs, preds)
    avg_sim_by_sent = similarity_by_sent.mean()
    ref_similarity_by_sent = wieting_sim(args, refs, preds)
    ref_avg_sim_by_sent = ref_similarity_by_sent.mean()
    # print('\n')
    # print(avg_sim_by_sent)
    char_ppl_by_sent = calc_flair_ppl(preds, aggregate=False)
    char_ppl = np.mean(char_ppl_by_sent)
    # print('\n')
    # print(char_ppl)
    token_ppl_by_sent = calc_gpt_ppl(preds, aggregate=False)
    token_ppl = np.mean(token_ppl_by_sent)
    # print('\n')
    # print(token_ppl)

    # print('\n')
    # print(cola_acc)
    gm = get_gm(args, accuracy, emb_sim, char_ppl)
    # print(gm)
    joint = get_j(args, accuracy_by_sent, similarity_by_sent, cola_stats, preds)
    # print(joint)
    results = {}
    results['model'] = name
    results['ACC'] = format_number(accuracy)
    results['EMB_SIM'] = format_number(emb_sim)
    results['ref_EMB_SIM'] = format_number(ref_emb_sim)
    results['SIM'] = format_number(avg_sim_by_sent)
    results['ref_SIM'] = format_number(ref_avg_sim_by_sent)
    results['BLEU'] = format_number(bleu)
    results['ref_BLEU'] = format_number(ref_bleu)
    results['CharPPL'] = format_number(char_ppl)
    results['TokenPPL'] = format_number(token_ppl)
    results['FL'] = format_number(cola_acc)
    results['GM'] = format_number(gm)
    results['J'] = format_number(joint)
    # blue, acc, sim, flu, J
    out = np.array([bleu, accuracy, avg_sim_by_sent, cola_acc, joint])

    per_sent_results = {}

    per_sent_results['inputs'] = inputs
    per_sent_results['preds'] = preds
    per_sent_results['refs'] = refs
    per_sent_results['ACC'] = [format_number(x) for x in accuracy_by_sent]
    results['ACC_CI'] = format_number(mean_confidence_interval(accuracy_by_sent, confidence=confidence))
    per_sent_results['EMB_SIM'] = [format_number(x) for x in emb_sim_stats]
    per_sent_results['ref_EMB_SIM'] = [format_number(x) for x in ref_emb_sim_stats]
    results['EMB_SIM_CI'] = format_number(mean_confidence_interval(emb_sim_stats, confidence=confidence))
    results['ref_EMB_SIM_CI'] = format_number(mean_confidence_interval(ref_emb_sim_stats, confidence=confidence))

    per_sent_results['SIM'] = [format_number(x) for x in similarity_by_sent]
    per_sent_results['ref_SIM'] = [format_number(x) for x in ref_similarity_by_sent]
    results['SIM_CI'] = format_number(mean_confidence_interval(similarity_by_sent, confidence=confidence))
    results['ref_SIM_CI'] = format_number(mean_confidence_interval(ref_similarity_by_sent, confidence=confidence))
    per_sent_results['BLEU'] = [format_number(x) for x in bleus]
    per_sent_results['ref_BLEU'] = [format_number(x) for x in ref_bleus]
    results['BLEU_CI'] = format_number(mean_confidence_interval(bleus, confidence=confidence))
    results['ref_BLEU_CI'] = format_number(mean_confidence_interval(ref_bleus, confidence=confidence))
    per_sent_results['CharPPL'] = [format_number(x) for x in char_ppl_by_sent]
    results['CharPPL_CI'] = format_number(mean_confidence_interval(char_ppl_by_sent, confidence=confidence))
    per_sent_results['TokenPPL'] = [format_number(x) for x in token_ppl_by_sent]
    results['TokenPPL_CI'] = format_number(mean_confidence_interval(token_ppl_by_sent, confidence=confidence))
    per_sent_results['FL'] = [format_number(x) for x in cola_stats]
    results['FL_CI'] = format_number(mean_confidence_interval(cola_stats, confidence=confidence))
    # per_sent_results['GM'] = "{:.4f}".format(gm)
    # per_sent_results['J'] = "{:.4f}".format(joint)
    # for key in per_sent_results:
    #     print(key, len(per_sent_results[key]))
    if per_sent_file_name is not None:
        df2 = pd.DataFrame(per_sent_results, index=list(range(len(per_sent_results['inputs']))))
        df2.to_csv(per_sent_file_name)
    else:
        df2 = None

    df = pd.DataFrame(results, index=[0])
    if save_path is not None:
        df.to_csv(save_path)

    return df, df2, out

def eval_json(json_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(json_path, "r", encoding='utf-8') as f: 
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
    #print("res", len(res), "input", len(toxic_sentences))
    assert len(res) == len(toxic_sentences), "length of preds and inputs dont match"
    # save_path = './'
    name_to_add = ''
    df, df2, out = evaluate(toxic_sentences, res, refs, name='diffuseq',
                       save_path=os.path.join(save_path, name_to_add + 'eval.csv'),
                       per_sent_file_name=os.path.join(save_path, name_to_add + 'per_sent.csv'))
    return out

import re
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

if __name__ == '__main__':
    #p = ['results/cf_000_mean_0.0', 'results/cf_000_mean_0.3', 'results/cf_000_mean_1.0']
    p = ['results/wiki']#, 'results/cf_000_default_5.0']
    for path in p:
        folders = os.listdir(path)
        folders = sorted(folders, key=natural_key)

        collect = []
        for i, f in enumerate(folders):
            sub_path = os.path.join(path, f)
            j = [a for a in os.listdir(sub_path) if '.json' in a]
            assert len(j) == 1

            json_path = os.path.join(sub_path, j[0])
            save_path = sub_path
            print(json_path)
            out = eval_json(json_path, save_path)
            print(5*'\n', out, 5*'\n')
            collect.append(out)

        print('ACC','SIM','FL','J') 
        for c in collect: print(c)
        s = np.array(collect)
        print(s)
        np.save(os.path.join(path, 'metrics.npy'), s)
