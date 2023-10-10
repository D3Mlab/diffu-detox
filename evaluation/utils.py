import numpy as np
import scipy.stats
import pandas as pd
import os
import json

def get_file_path(dirName):
    '''
    :param dirName: direction of the directory of all data
    :return: a list of file directions in that directory
    '''
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    # print('listoffiles',listOfFiles)
    return listOfFiles





def get_per_exp(list_of_files):

    results_id_address = {}

    for elem in list_of_files:
      decoding = 'random'
      lm = 'OPT'
      size = '1.3b'
      temperature = '1'

      if elem.endswith('eval.csv'):
        

        #decoding
        a = elem
        #top_k top_p
        if a.split('/')[-2].endswith('top_k_top_p'):

          decoding = 'top_k_top_p'

        elif a.split('/')[-2].endswith('top_p'):

          decoding = 'top_p'

        elif a.split('/')[-2].endswith('top_k'): 

          decoding = 'top_k'

        elif a.split('/')[-2].split('_')[-2] =='temp':

          decoding = a.split('/')[-2].split('_')[-3]
          temperature = a.split('/')[-2].split('_')[-1]

        elif a.split('/')[-2].split('_')[2] =='greedy' :

          decoding = "greedy"

        else:

          decoding = "random"



        lm = a.split('/')[-2].split('_')[0]
        size = a.split('/')[-2].split('_')[1]


        results_id_address[(lm,size,decoding,temperature)] = elem

    return results_id_address

def get_results_df(results_dir, output_csv):
  #Get all the pathes
  listOfFiles = get_file_path(results_dir)
  #Get all the experieces in a dictionary
  exp_addresses = get_per_exp(listOfFiles)

  #Make the base df to add all the experiments to
  base_exp = exp_addresses.popitem()
  base_df = pd.read_csv(base_exp[1])
  base_df['model'] = base_exp[0][0]
  base_df = base_df.drop(['Unnamed: 0'], axis=1)
  base_df['size'] = base_exp[0][1]
  base_df['decoding'] = base_exp[0][2]
  base_df['temperature'] = base_exp[0][3]

  for exp in exp_addresses:

    temp_df = pd.read_csv(exp_addresses[exp])
    temp_df['model'] = exp[0]
    temp_df = temp_df.drop(['Unnamed: 0'], axis=1)
    temp_df['size'] = exp[1]
    temp_df['decoding'] = exp[2]
    
    temp_df['temperature'] = exp[3]
    base_df = pd.concat([base_df,temp_df])
  base_df.to_csv(output_csv)
  return base_df



def get_profanity_list(addr='datasets/paradetox/profanity.txt'):
    words = []
    with open(addr, mode='r', encoding='utf-8') as f:
        words = f.readlines()
    return [w.replace('\n','').lower() for w in words]


def has_profanity(sentence, profanity_list):
    # print(profanity_list)
    s = sentence.lower()
    for word in profanity_list:
        # print(word)
        if s.find(word)!=-1:
            # print('yes')
            return True
    return False


def clean_curse(output_list, curse_list):
    output_curse_flag = [False] * len(output_list)
    for i, item in enumerate(output_list):
        word_start = ""
        for word in item.split():
            if word not in curse_list:
                word_start = word_start + " " + word
                output_curse_flag[i] = True
        output_list[i] = word_start
    return output_list, output_curse_flag