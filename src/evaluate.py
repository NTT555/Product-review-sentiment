import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB


def calculate_accuracy(y_true, y_pred):
    """Tính Accuracy"""
    return accuracy_score(y_true, y_pred)


def calculate_precision(y_true, y_pred, average='weighted'):
    """Tính Precision"""
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def calculate_recall(y_true, y_pred, average='weighted'):
    """Tính Recall"""
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def calculate_f1(y_true, y_pred, average='weighted'):
    """Tính F1-Score"""
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """
    Vẽ và lưu Confusion Matrix
    Args:
        y_true: Nhãn thật
        y_pred: Nhãn dự đoán
        labels: Danh sách nhãn (tên class)
        save_path: Đường dẫn lưu file ảnh (nếu None chỉ hiển thị)
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ Đã lưu confusion matrix tại: {save_path}")
    plt.show()


def evaluate_model(y_true, y_pred, labels=None, save_cm_path=None):
    """
    Đánh giá mô hình và in ra các chỉ số
    Args:
        y_true: Nhãn thật
        y_pred: Nhãn dự đoán
        labels: Tên các nhãn
        save_cm_path: Đường dẫn lưu ảnh confusion matrix
    Returns:
        dict metrics
    """
    acc = calculate_accuracy(y_true, y_pred)
    prec = calculate_precision(y_true, y_pred)
    rec = calculate_recall(y_true, y_pred)
    f1 = calculate_f1(y_true, y_pred)

    print("="*50)
    print("📊 BÁO CÁO ĐÁNH GIÁ MÔ HÌNH")
    print("="*50)
    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall   : {rec:.4f} ({rec*100:.2f}%)")
    print(f"F1-Score : {f1:.4f} ({f1*100:.2f}%)")
    print("\n=== Classification Report chi tiết ===")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

    plot_confusion_matrix(y_true, y_pred, labels=labels, save_path=save_cm_path)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }


def optimize_alpha_gridsearch(X_train, y_train, cv=5):
    """
    Dùng GridSearchCV để tối ưu tham số alpha cho MultinomialNB
    Args:
        X_train: Ma trận đặc trưng tập train
        y_train: Nhãn tập train
        cv: số lượng cross-validation folds
    Returns:
        best_model: Mô hình tốt nhất sau tối ưu
        best_params: Tham số alpha tốt nhất
        grid_result : Kết quả GridSearchCV
    """
    print("\n== Bắt đầu Grid Search để tối ưu alpha ==")

    model = MultinomialNB()
    param_grid = {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}

    grid_search = GridSearchCV(model,
                               param_grid,
                               scoring='f1_weighted',
                               cv=cv,
                               n_jobs=-1,
                               verbose=1)

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"✅ Kết quả Grid Search:\n  Best alpha: {best_params['alpha']}\n  Best F1-score (CV): {best_score:.4f}")
    print("Chi tiết điểm từng alpha:")
    results_df = pd.DataFrame(grid_search.cv_results_)
    for i, row in results_df.iterrows():
        print(f"  Alpha={row['param_alpha']:>6} | Mean F1-score: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")

    return best_model, best_params, grid_search


def run_evaluation(model, X_test, y_test, labels=None, save_cm_path=None):
    """
    Chạy đánh giá mô hình trên tập test
    Args:
        model: mô hình đã train
        X_test: dữ liệu test
        y_test: nhãn test
        labels: tên nhãn
        save_cm_path: đường dẫn lưu Confusion Matrix
    Returns:
        dict các metrics
    """
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, labels=labels, save_cm_path=save_cm_path)
    return metrics


if __name__ == "__main__":
    # Ví dụ đơn giản chạy thử code (không bắt buộc)
    print("📝 Ví dụ chạy đánh giá module evaluate.py")

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)

    labels = ['Negative', 'Positive']
    metrics = run_evaluation(model, X_test, y_test, labels=labels, save_cm_path='models/confusion_matrix.png')

    print("✅ Đánh giá xong.")
    print(metrics)
