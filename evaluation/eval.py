from evaluation.evaluation_detox.metric_tools.content_similarity import calc_bleu, calc_bleus, new_wieting_sim
from evaluation.evaluation_detox.metric_tools.fluency import  new_do_cola_eval
from evaluation.evaluation_detox.metric_tools.joint_metrics import *
import os
from os import listdir
from os.path import isfile, join

import pandas as pd
from evaluation.evaluation_detox.metric_tools.style_transfer_accuracy import classify_preds
import numpy as np
from box import Box
import json
import scipy


def format_number(num):
    return "{:.4f}".format(num)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h

def nontox_at_alpha(nontox, sim,fl, alpha):
    if sim>=alpha and fl >=alpha:
        return nontox
    else:
        return 0.0



def evaluate(inputs, preds, refs, name='test', save_path=None, per_sent_file_name=None, confidence=0.95):
    args = Box({
        'batch_size': 16,
        'cola_classifier_path': './',  # TODO: put your path here
        'wieting_tokenizer_path': 'sim.sp.30k.model',
        'wieting_model_path': 'sim.pt',
        't1': 75.,  # this is default value
        't2': 70.,  # this is default value
        't3': 12.  # this is default value
    })
    # print(len(preds))
    accuracy_by_sent,nontox_probs_by_sent = classify_preds(args, preds)

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
    # emb_sim_stats = flair_sim(args, inputs, preds)
    # emb_sim = emb_sim_stats.mean()
    # ref_emb_sim_stats = flair_sim(args, refs, preds)
    # ref_emb_sim = ref_emb_sim_stats.mean()
    # # print('\n')
    # print(emb_sim)
    # new_wieting_sim

    similarity_by_sent = new_wieting_sim(args, inputs, preds)
    avg_sim_by_sent = similarity_by_sent.mean()
    ref_similarity_by_sent = new_wieting_sim(args, refs, preds)
    ref_avg_sim_by_sent = ref_similarity_by_sent.mean()
    # print('\n')
    # print(avg_sim_by_sent)
    # char_ppl_by_sent = calc_flair_ppl(preds, aggregate=False)
    # char_ppl = np.mean(char_ppl_by_sent)
    # # print('\n')
    # # print(char_ppl)
    # token_ppl_by_sent = calc_gpt_ppl(preds, aggregate=False)
    # token_ppl = np.mean(token_ppl_by_sent)
    # print('\n')
    # print(token_ppl)
    # new_do_cola_eval
   
    cola_stats,fl_probs_by_sent = new_do_cola_eval(args, preds)
    cola_acc = sum(cola_stats) / len(preds)


    nontox_at_alpha_values_50 = [nontox_at_alpha(nontox_probs_by_sent[i],similarity_by_sent[i],fl_probs_by_sent[i],0.5) for i in range(len(fl_probs_by_sent))]
    nontox_at_alpha_values_55 = [nontox_at_alpha(nontox_probs_by_sent[i],similarity_by_sent[i],fl_probs_by_sent[i],0.55) for i in range(len(fl_probs_by_sent))]
    nontox_at_alpha_values_60 = [nontox_at_alpha(nontox_probs_by_sent[i],similarity_by_sent[i],fl_probs_by_sent[i],0.6) for i in range(len(fl_probs_by_sent))]
    nontox_at_alpha_values_65 = [nontox_at_alpha(nontox_probs_by_sent[i],similarity_by_sent[i],fl_probs_by_sent[i],0.65) for i in range(len(fl_probs_by_sent))]
    nontox_at_alpha_values_70 = [nontox_at_alpha(nontox_probs_by_sent[i],similarity_by_sent[i],fl_probs_by_sent[i],0.7) for i in range(len(fl_probs_by_sent))]
    nontox_at_alpha_values_75 = [nontox_at_alpha(nontox_probs_by_sent[i],similarity_by_sent[i],fl_probs_by_sent[i],0.75) for i in range(len(fl_probs_by_sent))]
    nontox_at_alpha_values_80 = [nontox_at_alpha(nontox_probs_by_sent[i],similarity_by_sent[i],fl_probs_by_sent[i],0.8) for i in range(len(fl_probs_by_sent))]
    nontox_at_alpha_values_85 = [nontox_at_alpha(nontox_probs_by_sent[i],similarity_by_sent[i],fl_probs_by_sent[i],0.85) for i in range(len(fl_probs_by_sent))]
    nontox_at_alpha_values_90 = [nontox_at_alpha(nontox_probs_by_sent[i],similarity_by_sent[i],fl_probs_by_sent[i],0.9) for i in range(len(fl_probs_by_sent))]
    nontox_at_alpha_values_95 = [nontox_at_alpha(nontox_probs_by_sent[i],similarity_by_sent[i],fl_probs_by_sent[i],0.95) for i in range(len(fl_probs_by_sent))]
    
    # print('\n')
    # print(cola_acc)
    # gm = get_gm(args, accuracy, emb_sim, char_ppl)
    # print(gm)
    joints = get_js(args, accuracy_by_sent, similarity_by_sent, cola_stats, preds)
    joint = get_j(args, accuracy_by_sent, similarity_by_sent, cola_stats, preds)
    assert np.mean(joints) == joint
    # print(joint)
    results = {}
    results['model'] = name
    results['STA'] = format_number(accuracy)
    STA_prob = np.mean(nontox_probs_by_sent)
    results['STA_prob'] = format_number(STA_prob)
    # results['EMB_SIM'] = format_number(emb_sim)
    # results['ref_EMB_SIM'] = format_number(ref_emb_sim)
    
    results['SIM'] = format_number(avg_sim_by_sent)
    results['ref_SIM'] = format_number(ref_avg_sim_by_sent)

    # results['CharPPL'] = format_number(char_ppl)
    # results['TokenPPL'] = format_number(token_ppl)
    results['FL'] = format_number(cola_acc)
    FL_prob = np.mean(fl_probs_by_sent)
    results['FL_prob'] = format_number(FL_prob)
    results['J'] = format_number(joint)
    scalarizations = [[.33, .33, .33], [0.5, .25, .25], [0.25, .5, .25], [0.25, .25, .5]]
    # print(avg_sim_by_sent,FL_prob,STA_prob)
    for c in scalarizations:
        # print(c)
        results[str(c)] = c[0] * STA_prob + c[1] * avg_sim_by_sent + c[2] * FL_prob

    # results['GM'] = format_number(gm)
    results['BLEU'] = format_number(bleu)
    results['ref_BLEU'] = format_number(ref_bleu)

    results['nontox@0.5'] = np.mean(nontox_at_alpha_values_50)
    results['nontox@0.55'] = np.mean(nontox_at_alpha_values_55)
    results['nontox@0.6'] = np.mean(nontox_at_alpha_values_60)
    results['nontox@0.65'] = np.mean(nontox_at_alpha_values_65)
    results['nontox@0.7'] = np.mean(nontox_at_alpha_values_70)
    results['nontox@0.75'] = np.mean(nontox_at_alpha_values_75)
    results['nontox@0.8'] = np.mean(nontox_at_alpha_values_80)
    results['nontox@0.85'] = np.mean(nontox_at_alpha_values_85)
    results['nontox@0.9'] = np.mean(nontox_at_alpha_values_90)
    results['nontox@0.95'] = np.mean(nontox_at_alpha_values_95)

    per_sent_results = {}

    per_sent_results['nontox@0.5'] = nontox_at_alpha_values_50
    per_sent_results['nontox@0.55'] = nontox_at_alpha_values_55
    per_sent_results['nontox@0.6'] = nontox_at_alpha_values_60
    per_sent_results['nontox@0.65'] = nontox_at_alpha_values_65
    per_sent_results['nontox@0.7'] = nontox_at_alpha_values_70
    per_sent_results['nontox@0.75'] = nontox_at_alpha_values_75
    per_sent_results['nontox@0.8'] = nontox_at_alpha_values_80
    per_sent_results['nontox@0.85'] = nontox_at_alpha_values_85
    per_sent_results['nontox@0.9'] = nontox_at_alpha_values_90
    per_sent_results['nontox@0.95'] = nontox_at_alpha_values_95


    per_sent_results['inputs'] = inputs
    per_sent_results['preds'] = preds
    per_sent_results['refs'] = refs
    per_sent_results['STA'] = [format_number(x) for x in accuracy_by_sent]
    per_sent_results['STA_prob'] = [format_number(x) for x in nontox_probs_by_sent]

    results['STA_CI'] = format_number(mean_confidence_interval(accuracy_by_sent, confidence=confidence))
    # per_sent_results['EMB_SIM'] = [format_number(x) for x in emb_sim_stats]
    # per_sent_results['ref_EMB_SIM'] = [format_number(x) for x in ref_emb_sim_stats]
    # results['EMB_SIM_CI'] = format_number(mean_confidence_interval(emb_sim_stats, confidence=confidence))
    # results['ref_EMB_SIM_CI'] = format_number(mean_confidence_interval(ref_emb_sim_stats, confidence=confidence))

    per_sent_results['SIM'] = [format_number(x) for x in similarity_by_sent]
    per_sent_results['ref_SIM'] = [format_number(x) for x in ref_similarity_by_sent]
    results['SIM_CI'] = format_number(mean_confidence_interval(similarity_by_sent, confidence=confidence))
    results['ref_SIM_CI'] = format_number(mean_confidence_interval(ref_similarity_by_sent, confidence=confidence))
    per_sent_results['BLEU'] = [format_number(x) for x in bleus]
    per_sent_results['ref_BLEU'] = [format_number(x) for x in ref_bleus]
    results['BLEU_CI'] = format_number(mean_confidence_interval(bleus, confidence=confidence))
    results['ref_BLEU_CI'] = format_number(mean_confidence_interval(ref_bleus, confidence=confidence))
    # per_sent_results['CharPPL'] = [format_number(x) for x in char_ppl_by_sent]
    # results['CharPPL_CI'] = format_number(mean_confidence_interval(char_ppl_by_sent, confidence=confidence))
    # per_sent_results['TokenPPL'] = [format_number(x) for x in token_ppl_by_sent]
    # results['TokenPPL_CI'] = format_number(mean_confidence_interval(token_ppl_by_sent, confidence=confidence))
    per_sent_results['FL'] = [format_number(x) for x in cola_stats]
    per_sent_results['FL_prob'] = [format_number(x) for x in fl_probs_by_sent]

    results['FL_CI'] = format_number(mean_confidence_interval(cola_stats, confidence=confidence))
    # per_sent_results['GM'] = "{:.4f}".format(gm)
    per_sent_results['J'] = [format_number(x) for x in joints]
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



    return df, df2
















