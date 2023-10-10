import torch
import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu


from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer


try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False



def calc_bleu(inputs, preds):
    bleu_sim = 0
    counter = 0
    # print('Calculating BLEU similarity')
    for i in range(len(inputs)):
        if len(inputs[i]) > 3 and len(preds[i]) > 3:
            bleu_sim += sentence_bleu([inputs[i]], preds[i])
            counter += 1

    return float(bleu_sim / counter)

def calc_bleus(inputs, preds):
    bleu_sim = 0
    counter = 0
    blue_sims = []
    # print('Calculating BLEU similarity')
    for i in range(len(inputs)):
        # if len(inputs[i]) > 3 and len(preds[i]) > 3:
        bleu_sim = sentence_bleu([inputs[i]], preds[i])
        # print(bleu_sim)
        counter += 1
        blue_sims.append(bleu_sim)
        # else:
        #     blue_sims.append(0)

    return blue_sims











def new_wieting_sim(args, inputs, preds):
    assert len(inputs) == len(preds)
    # print('Calculating similarity by Wieting subword-embedding SIM model')

    # sim_model = SimilarityEvaluator(args.wieting_model_path, args.wieting_tokenizer_path)
    model = SentenceTransformer('sentence-transformers/LaBSE')

    inputs_embeddings = model.encode(inputs)
    preds_embeddings = model.encode(preds)
    sim_scores = []

    for i in tqdm.tqdm(range(len(inputs))):
        sim_scores.append(np.dot(inputs_embeddings[i,:],preds_embeddings[i,:]))
    return np.array(sim_scores)