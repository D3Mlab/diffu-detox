from detox_experiment import fewshot_experiment, evaluate_results



# seeds = [200, 300, 400, 500]
seeds = [300]
language_models = [('BLOOM', '1b1'), ('OPT', '1.3b'), ('GPT2', 'large'), ('BLOOM', '560m')]
# language_models = [('OPT', '1.3b')]
for seed in seeds:
    for lm in language_models:
        # fewshot_experiment('detox_exp/LM_looping_' + str(seed) + '/' + lm[0] + "_" + lm[1] + '_greedy', do_sample=False,
        #                    model_name=lm[0],
        #                    model_size=lm[1], seed=seed)
        evaluate_results('detox_exp/LM_looping_' + str(seed) + '/' + lm[0] + "_" + lm[1] + '_greedy')


        # fewshot_experiment('detox_exp/LM_looping_' + str(seed) + '/' + lm[0] + "_" + lm[1] + '_top_p', do_sample=True,
        #                    top_k=None,
        #                    model_name=lm[0],
        #                    model_size=lm[1], seed=seed)
        evaluate_results('detox_exp/LM_looping_' + str(seed) + '/' + lm[0] + "_" + lm[1] + '_top_p')


        # fewshot_experiment('detox_exp/LM_looping_' + str(seed) + '/' + lm[0] + "_" + lm[1] + '_top_k', do_sample=True,
        #                    top_p=None,
        #                    model_name=lm[0],
        #                    model_size=lm[1], seed=seed)
        evaluate_results('detox_exp/LM_looping_' + str(seed) + '/' + lm[0] + "_" + lm[1] + '_top_k')


        # fewshot_experiment('detox_exp/LM_looping_' + str(seed) + '/' + lm[0] + "_" + lm[1] + '_top_k_top_p',
        #                    do_sample=True,
        #                    model_name=lm[0],
        #                    model_size=lm[1], seed=seed)
        evaluate_results('detox_exp/LM_looping_' + str(seed) + '/' + lm[0] + "_" + lm[1] + '_top_k_top_p')


        # fewshot_experiment('detox_exp/LM_looping_' + str(seed) + '/' + lm[0] + "_" + lm[1] + '_random_sample',
        #                    do_sample=True, top_p=None,
        #                    top_k=None,
        #                    model_name=lm[0], model_size=lm[1], seed=seed)
        evaluate_results('detox_exp/LM_looping_' + str(seed) + '/' + lm[0] + "_" + lm[1] + '_random_sample')


        # fewshot_experiment('detox_exp/LM_looping_' + str(seed) + '/' + lm[0] + "_" + lm[1] + '_random_temp_5',
        #                    do_sample=True, top_p=None,
        #                    top_k=None,
        #                    temperature=5.0,
        #                    model_name=lm[0], model_size=lm[1], seed=100)
        evaluate_results('detox_exp/LM_looping_' + str(seed) + '/' + lm[0] + "_" + lm[1] + '_random_temp_5')



        # fewshot_experiment('detox_exp/LM_looping_' + str(seed) + '/' + lm[0] + "_" + lm[1] + '_random_temp_10',
        #                    do_sample=True,
        #                    top_p=None,
        #                    top_k=None,
        #                    temperature=10.0, model_name=lm[0], model_size=lm[1], seed=seed)
        evaluate_results('detox_exp/LM_looping_' + str(seed) + '/' + lm[0] + "_" + lm[1] + '_random_temp_10')


        # fewshot_experiment('detox_exp/LM_looping_' + str(seed) + '/' + lm[0] + "_" + lm[1] + '_random_temp_0.1',
        #                    do_sample=True,
        #                    top_p=None,
        #                    top_k=None,
        #                    temperature=0.1, model_name=lm[0], model_size=lm[1], seed=seed)
        evaluate_results('detox_exp/LM_looping_' + str(seed) + '/' + lm[0] + "_" + lm[1] + '_random_temp_0.1')


        # fewshot_experiment('detox_exp/LM_looping_' + str(seed) + '/' + lm[0] + "_" + lm[1] + '_random_temp_0.2',
        #                    do_sample=True,
        #                    top_p=None,
        #                    top_k=None,
        #                    temperature=0.2, model_name=lm[0], model_size=lm[1], seed=seed)
        evaluate_results('detox_exp/LM_looping_' + str(seed) + '/' + lm[0] + "_" + lm[1] + '_random_temp_0.2')



# # evaluate_results('detox_exp/LM_looping/' + lm[0] + "_" + lm[1] + '_random_temp_0.1')
# evaluate_results('detox_exp/LM_looping/' + lm[0] + "_" + lm[1] + '_random_temp_0.2')
# evaluate_results('detox_exp/LM_looping/' + lm[0] + "_" + lm[1] + '_greedy')


from style_paraphrase.evaluation.detox.process import get_results_df

df = get_results_df('detox_exp/LM_looping_300/', 'output_300.csv')
print(len(df))