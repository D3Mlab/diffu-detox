from detox_experiment import fewshot_experiment,zeroshot_experiment,eval_json

seed =100
# zeroshot_experiment('detox_exp/TK_ZS_greedy', do_sample=False, model_name="TK_FS",model_size="large",
# task_def=
# "Definition: Convert an offensive sentence to a polite sentence. An offensive sentence has offensive words and swear words such as fuck. A polite sentence does not has any offensive words and it has nice words. -"
# )


# fewshot_experiment('detox_exp/TK_FS_greedy', model_name="TK_FS",model_size="large",k=4,
# task_def=
# "Definition: Convert an offensive sentence to a polite sentence. -"
# )
# print the whole prompt in the end


eval_json('deffuseq.json','diffuseq_res/')
