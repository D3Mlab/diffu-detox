import os
import numpy as np
import math
import torch
import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, RobertaTokenizer, RobertaForSequenceClassification
from fairseq.models.roberta import RobertaModel
from flair.embeddings import FlairEmbeddings
from fairseq.data.data_utils import collate_tokens
import torch.nn.functional as F







def new_do_cola_eval(args, preds, soft=False):
    results = []
    probs = []

    tokenizer = RobertaTokenizer.from_pretrained('cointegrated/roberta-large-cola-krishna2020')
    model = RobertaForSequenceClassification.from_pretrained('cointegrated/roberta-large-cola-krishna2020').cuda()

    for i in tqdm.tqdm(range(0, len(preds), args.batch_size)):
        batch = tokenizer(preds[i:i + args.batch_size], return_tensors='pt', padding=True)
        for key in batch.keys():
            batch[key] = batch[key].to('cuda')
        with torch.no_grad():
            outputs = model(**batch)['logits']
        result = outputs.argmax(1).float().data.tolist()
        results.extend([1 - item for item in result])
        # results.extend(result)

        prob = F.softmax(outputs, dim=1).float().cpu().numpy()
        prob = prob[:,0]
        probs.extend(prob)
    return results, probs