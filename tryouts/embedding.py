from transformers import BertModel, BertTokenizer
import torch
import spacy
import re
import pandas as pd
import faiss
import numpy as np
import os
nlp = spacy.load('en_core_web_sm')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')



def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text) 
    doc = nlp(str(text))
    preprocessed_text = []
    for token in doc:
        if token.is_punct or token.like_num or token.is_space:
            continue
        preprocessed_text.append(token.lemma_.lower().strip())
    return ' '.join(preprocessed_text)

def generate_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embedding

