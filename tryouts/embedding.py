from transformers import BertModel, BertTokenizer
import torch
import spacy
import re
import pandas as pd
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
nlp = spacy.load('en_core_web_sm')

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

model = SentenceTransformer('all-MiniLM-L6-v2')




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


def process_text(input_text):
    cleaned_text = re.sub(r'[^A-Za-z\s]', '', input_text)
    
    words = cleaned_text.split()
    
    processed_words = [word.capitalize() if word.isupper() else word for word in words]
    
    return ' '.join(processed_words)

