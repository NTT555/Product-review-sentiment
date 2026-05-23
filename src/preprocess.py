import re
import pandas as pd
from underthesea import word_tokenize

def clean_text(text):
    if not isinstance(text, str):
        return ""
     
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text) 
    return text.strip()

def tokenize_text(text):
    return word_tokenize(text, format="text")

stopwords_vi = ["là", "và", "của", "có", "cho"]
stopwords_en = ["the", "is", "and", "in", "on"]

def remove_stopwords(text):
    words = text.split()
    filtered = [w for w in words if w not in stopwords_vi and w not in stopwords_en]
    return " ".join(filtered)


def preprocess_pipeline(text):
    text = clean_text(text)
    text = tokenize_text(text)
    text = remove_stopwords(text)

    if text == "":
        return "empty"   

    return text

def load_and_preprocess(path="data/processed/clean_data.csv"):
    df = pd.read_csv(path)
    
    df["clean_text"] = df["text"].apply(preprocess_pipeline)
    
    return df