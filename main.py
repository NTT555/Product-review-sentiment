import pandas as pd
from src.data_loader import load_data
from src.preprocess import clean_text
from src.features import run_feature_pipeline
from src.train import train_mnb_model
from src.evaluate import evaluate_model

def main():
    print("🚀 BẮT ĐẦU PIPELINE HUẤN LUYỆN LẠI MÔ HÌNH\n")
    
    # 1. Đọc dữ liệu đã gán nhãn thủ công
    data_path = 'data/raw/labeled_comments.xlsx' 
    df = load_data(data_path)
    
    # 2. Tiền xử lý văn bản thô
    print("⏳ Đang thực hiện làm sạch và tách từ văn bản...")
    df['clean_text'] = df['Text'].apply(clean_text)
    
    # 3. Trích xuất đặc trưng và chia tập dữ liệu
    print("⏳ Đang phân tách Train/Test và vector hóa ma trận TF-IDF...")
    feature_dict = run_feature_pipeline(df)
    
    # Giải nén các biến từ kết quả trả về
    X_train_vec = feature_dict["X_train_tfidf"]
    X_test_vec  = feature_dict["X_test_tfidf"]
    y_train     = feature_dict["y_train"]
    y_test      = feature_dict["y_test"]
    vectorizer  = feature_dict["vectorizer"]
    
    # 4. Huấn luyện mô hình Multinomial Naive Bayes
    model = train_mnb_model(X_train_vec, y_train)
    
    # 5. Đánh giá chất lượng mô hình
    evaluate_model(model, X_test_vec, y_test)


    # KIỂM THỬ THỰC TẾ (UNSEEN DATA TEST)
    print("\n--- KIỂM THỬ KHẢ NĂNG DỰ ĐOÁN THỰC TẾ ---")
    test_comments = [
        "Sản phẩm dùng cực kỳ mượt, giao hàng siêu nhanh, 10 điểm!",
        "Mới xài 2 ngày đã hỏng, quá tệ, sẽ không bao giờ mua lại.",
        "Dùng cũng tạm ổn nhưng pin hơi nhanh hết, giá này thì chấp nhận được.",
        "Sản phẩm không hề tốt",
    ]
    
    clean_test = [clean_text(c) for c in test_comments]
    test_vec = vectorizer.transform(clean_test)
    predictions = model.predict(test_vec)
    
    for c, p in zip(test_comments, predictions):
        print(f"Bình luận: {c}")
        print(f"Dự đoán  : [{p.upper()}]\n")

if __name__ == "__main__":
    main()