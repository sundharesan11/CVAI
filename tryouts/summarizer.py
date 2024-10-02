from transformers import AutoTokenizer, AutoModelWithLMHead

from transformers import pipeline

summarizer = pipeline("summarization", model="Falconsai/text_summarization")       


def summarize(text):
    l = len(text)
    if l<1000:
        maxlength = l/1.5
    else:
        maxlength = 1000
    summary = summarizer(text, max_length=maxlength, min_length=30, do_sample=False)                
    return summary[0]['summary_text']