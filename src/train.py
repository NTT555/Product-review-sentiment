from sklearn.naive_bayes import MultinomialNB


def train_model(X_train, y_train, X_test, y_test):

    # Khởi tạo model
    model = MultinomialNB(alpha=1.0)

    # Huấn luyện model
    model.fit(X_train, y_train)

    print("Training completed!")

    return model

