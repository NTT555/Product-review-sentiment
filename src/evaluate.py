from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, X_test_vec, y_test):
    print("\n" + "="*40)
    print(" BÁO CÁO ĐÁNH GIÁ MÔ HÌNH (EVALUATION)")
    print("="*40)
    
    y_pred = model.predict(X_test_vec)
    
    print(f"Độ chính xác tổng thể (Accuracy): {accuracy_score(y_test, y_pred):.4f}\n")
    print("Chi tiết Precision, Recall, F1-Score:")
    print(classification_report(y_test, y_pred))