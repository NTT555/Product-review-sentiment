import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

CONFIG = {
    "test_size": 0.20,
    "random_state": 42,
    "ngram_range": (1, 2),
    "max_features": 50_000,
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
    "analyzer": "word",
    "token_pattern": r"(?u)\b\w+\b",
    "output_dir": "data/processed",
    "vectorizer_path": "models/tfidf_vectorizer.pkl",
}


def split_data(df: pd.DataFrame, config: dict = CONFIG):
    X = df["text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def build_tfidf_features(X_train, X_test, config: dict = CONFIG):
    vectorizer = TfidfVectorizer(
        ngram_range=config["ngram_range"],
        max_features=config["max_features"],
        min_df=config["min_df"],
        max_df=config["max_df"],
        sublinear_tf=config["sublinear_tf"],
        analyzer=config["analyzer"],
        token_pattern=config["token_pattern"],
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer


def save_features(X_train_tfidf, X_test_tfidf,
                  y_train, y_test,
                  vectorizer, config: dict = CONFIG):
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(config["vectorizer_path"]), exist_ok=True)
    save_npz(os.path.join(config["output_dir"], "X_train.npz"), X_train_tfidf)
    save_npz(os.path.join(config["output_dir"], "X_test.npz"),  X_test_tfidf)
    np.save(os.path.join(config["output_dir"], "y_train.npy"), y_train.values)
    np.save(os.path.join(config["output_dir"], "y_test.npy"),  y_test.values)
    joblib.dump(vectorizer, config["vectorizer_path"])


def run_feature_pipeline(df: pd.DataFrame, config: dict = CONFIG) -> dict:
    X_train, X_test, y_train, y_test = split_data(df, config)
    X_tr, X_te, vect = build_tfidf_features(X_train, X_test, config)
    save_features(X_tr, X_te, y_train, y_test, vect, config)
    return {
        "X_train_tfidf": X_tr,
        "X_test_tfidf":  X_te,
        "y_train":       y_train,
        "y_test":        y_test,
        "vectorizer":    vect,
    }
