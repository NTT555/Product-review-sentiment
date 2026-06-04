import os
import math
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from scipy.sparse import save_npz, csr_matrix

#Config
CONFIG = {
    "test_size": 0.20,
    "random_state": 42,
    "ngram_range": (1, 2),
    "max_features": 50_000,
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
    "output_dir": "data/processed",
    "vectorizer_path": "models/tfidf_vectorizer.pkl",
}

#Train/Test Split
def train_test_split_custom(
    X, y,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify=None,
):
    np.random.seed(random_state)

    if stratify is not None:
        train_idx, test_idx = [], []
        for label in np.unique(stratify):
            idx = np.where(np.array(stratify) == label)[0]
            np.random.shuffle(idx)
            n_test = max(1, int(len(idx) * test_size))
            test_idx.extend(idx[:n_test])
            train_idx.extend(idx[n_test:])
    else:
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        n_test = int(len(X) * test_size)
        test_idx, train_idx = idx[:n_test], idx[n_test:]

    if hasattr(X, "iloc"):
        return (
            X.iloc[train_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[test_idx],
        )
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

#TF-IDF Vectorizer
class TfidfVectorizerCustom:
    def __init__(
        self,
        ngram_range: tuple = (1, 1),
        max_features: int = None,
        min_df: int = 1,
        max_df: float = 1.0,
        sublinear_tf: bool = False,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.vocabulary_: dict = {}
        self.idf_: dict = {}

    def _get_ngrams(self, tokens: list[str]) -> list[str]:
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            ngrams += [
                " ".join(tokens[i : i + n])
                for i in range(len(tokens) - n + 1)
            ]
        return ngrams

    def fit(self, docs: list[str]) -> "TfidfVectorizerCustom":
        N = len(docs)
        df_count = Counter()

        for doc in docs:
            df_count.update(set(self._get_ngrams(doc.split())))

        max_df_count = (
            int(self.max_df * N) if isinstance(self.max_df, float) else self.max_df
        )
        vocab = {
            t: c for t, c in df_count.items()
            if self.min_df <= c <= max_df_count
        }

        sorted_vocab = sorted(vocab.items(), key=lambda x: -x[1])
        if self.max_features:
            sorted_vocab = sorted_vocab[: self.max_features]

        self.vocabulary_ = {t: i for i, (t, _) in enumerate(sorted_vocab)}
        self.idf_ = {
            t: math.log((1 + N) / (1 + c)) + 1
            for t, c in dict(sorted_vocab).items()
        }
        return self

    def transform(self, docs: list[str]) -> csr_matrix:
        rows, cols, vals = [], [], []

        for i, doc in enumerate(docs):
            tf = Counter(self._get_ngrams(doc.split()))
            for term, count in tf.items():
                if term not in self.vocabulary_:
                    continue
                tf_val = (1 + math.log(count)) if self.sublinear_tf else count
                rows.append(i)
                cols.append(self.vocabulary_[term])
                vals.append(tf_val * self.idf_[term])

        mat = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(docs), len(self.vocabulary_)),
        )

        # L2 normalise
        norm = np.sqrt(mat.multiply(mat).sum(axis=1)).A1
        norm[norm == 0] = 1
        return mat.multiply(1 / norm[:, None])

    def fit_transform(self, docs: list[str]) -> csr_matrix:
        return self.fit(docs).transform(docs)

#Pipeline
def split_data(df: pd.DataFrame, config: dict = CONFIG):
    X, y = df["clean_text"], df["label"]
    return train_test_split_custom(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )


def build_tfidf_features(X_train, X_test, config: dict = CONFIG):
    vectorizer = TfidfVectorizerCustom(
        ngram_range=config["ngram_range"],
        max_features=config["max_features"],
        min_df=config["min_df"],
        max_df=config["max_df"],
        sublinear_tf=config["sublinear_tf"],
    )
    return vectorizer.fit_transform(X_train), vectorizer.transform(X_test), vectorizer


def save_features(
    X_train_tfidf, X_test_tfidf,
    y_train, y_test,
    vectorizer,
    config: dict = CONFIG,
):
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
