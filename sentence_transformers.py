from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np


df = pd.read_csv('next_phase/channel_messages_cleaned.csv')
messages = list(df['message'])
# print(type(messages))
sentences = []
for x in messages:
    sentences.append(str(x))

# Mean Pooling - Take attention mask into account for correct averaging


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
# sentences = ["Questo è un esempio di frase", "Questo è un ulteriore esempio"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('efederici/sentence-bert-base')
model = AutoModel.from_pretrained('efederici/sentence-bert-base')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True,
                          truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, mean pooling.
sentence_embeddings = mean_pooling(
    model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
# print(sentence_embeddings)
# print(sentence_embeddings.shape)

X = np.array(sentence_embeddings)
# print(X)
clustering = DBSCAN(eps=3, min_samples=2).fit(X)
print(clustering)
# now use this embeddings as a representation to cluster. Use clustering now. These embeddings gve us similarity.
