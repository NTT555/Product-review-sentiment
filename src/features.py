import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from preprocess import preprocess_pipeline

#Cấu hình trung tâm
CONFIG = {
    # Chia tập
    "test_size": 0.20,
    "random_state": 42,

    # TF-IDF
    "ngram_range": (1, 2),      # Unigram + Bigram
    "max_features": 50_000,
    "min_df": 2,                # Bỏ từ chỉ xuất hiện 1 lần
    "max_df": 0.95,             # Bỏ từ quá phổ biến
    "sublinear_tf": True,       # log(tf) — chuẩn hoá tốt hơn
    "analyzer": "word",
    "token_pattern": r"(?u)\b\w+\b",

    # Đường dẫn lưu
    "output_dir": "data/processed",
    "vectorizer_path": "models/tfidf_vectorizer.pkl",
}

#Chia Train / Test
def split_data(df: pd.DataFrame, config: dict = CONFIG):
    """
    Chia 80% Train / 20% Test.
    stratify=y — giữ tỉ lệ nhãn đều ở cả 2 tập.
    X_train, X_test : pd.Series  — văn bản đã xử lý
    y_train, y_test : pd.Series  — nhãn
    """
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )

    print(f"[split_data] Train: {len(X_train):,} mẫu | Test: {len(X_test):,} mẫu")
    print(f"[split_data] Nhãn Train:\n{y_train.value_counts().to_string()}")
    print(f"[split_data] Nhãn Test:\n{y_test.value_counts().to_string()}")

    return X_train, X_test, y_train, y_test

#TF-IDF
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

    X_train_tfidf = vectorizer.fit_transform(X_train)  # ← fit ở đây
    X_test_tfidf  = vectorizer.transform(X_test)        # ← KHÔNG fit

    print(f"[build_tfidf] Từ vựng : {len(vectorizer.vocabulary_):,} terms")
    print(f"[build_tfidf] Train   : {X_train_tfidf.shape}")
    print(f"[build_tfidf] Test    : {X_test_tfidf.shape}")

    return X_train_tfidf, X_test_tfidf, vectorizer

#Lưu artifacts
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

    print(f"[save_features] Đặc trưng → '{config['output_dir']}'")
    print(f"[save_features] Vectorizer → '{config['vectorizer_path']}'")

#Pipeline hoàn chỉnh
def run_feature_pipeline(df: pd.DataFrame, config: dict = CONFIG) -> dict:
    print("=" * 55)
    print("  FEATURE ENGINEERING — Thành viên 3")
    print("=" * 55)

    df_clean = apply_preprocessing(df)                        
    X_train, X_test, y_train, y_test = split_data(df_clean)   
    X_tr, X_te, vect = build_tfidf_features(X_train, X_test) 
    save_features(X_tr, X_te, y_train, y_test, vect)           

    print("=" * 55)
    print("  Xong! Sẵn sàng bàn giao cho Thành viên 4.")
    print("=" * 55)

    return {
        "X_train_tfidf": X_tr,
        "X_test_tfidf":  X_te,
        "y_train":       y_train,
        "y_test":        y_test,
        "vectorizer":    vect,
    }

#Validation nội bộ
def _validate_dataframe(df: pd.DataFrame):
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame thiếu cột: {missing}. Cần có: {{'text', 'label'}}")
    if df["text"].isnull().any():
        raise ValueError("Cột 'text' chứa giá trị null — TV1 cần xử lý trước.")
    if df["label"].isnull().any():
        raise ValueError("Cột 'label' chứa giá trị null — kiểm tra dữ liệu gốc.")
    print(f"[validate] OK — {len(df):,} dòng | nhãn: {sorted(df['label'].unique().tolist())}")
