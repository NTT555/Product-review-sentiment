from sklearn.naive_bayes import MultinomialNB

def train_mnb_model(X_train_vec, y_train):
    print("⏳ Đang huấn luyện mô hình Multinomial Naive Bayes...")
    
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_vec, y_train)
    
    print("✅ Huấn luyện hoàn tất!")
    return model