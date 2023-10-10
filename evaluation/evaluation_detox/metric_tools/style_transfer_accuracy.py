from transformers import RobertaTokenizer, RobertaForSequenceClassification
import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch

def classify_preds(args, preds):
    # print('Calculating style of predictions')
    results = []

    tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
    model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier').cuda()
    probs = []
    for i in tqdm.tqdm(range(0, len(preds), args.batch_size)):
        batch = tokenizer(preds[i:i + args.batch_size], return_tensors='pt', padding=True)
        for key in batch.keys():
            batch[key] = batch[key].to('cuda')
        with torch.no_grad():
            outputs = model(**batch)['logits']
        result = outputs.argmax(1).float().data.tolist()
        results.extend([1 - item for item in result])
        
        prob = F.softmax(outputs, dim=1).float().cpu().numpy()
        prob = prob[:,0]
        probs.extend(prob)

    return results, probs