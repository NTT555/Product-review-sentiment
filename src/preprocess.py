import re
from underthesea import word_tokenize

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def tokenize_text(text):
    return word_tokenize(text)

stopwords_vi = ["là", "và", "của", "có", "cho"]
stopwords_en = ["the", "is", "and", "in", "on"]

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stopwords_vi and word not in stopwords_en]