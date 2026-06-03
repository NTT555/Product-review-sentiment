import numpy as np


class MyMultinomialNB:

    def __init__(self, alpha=1.0):
        # Hệ số Laplace smoothing
        self.alpha = alpha

    def fit(self, X_train, y_train):

       
        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()

        y_train = np.array(y_train)

        # Các lớp nhãn
        self.classes = np.unique(y_train)

        # Số lượng đặc trưng
        n_features = X_train.shape[1]

        # Lưu P(class)
        self.class_prior = {}

        # Lưu P(word|class)
        self.feature_prob = {}

        total_samples = len(y_train)

        for c in self.classes:

            # Lấy các mẫu thuộc lớp c
            X_c = X_train[y_train == c]

            # Tính xác suất tiên nghiệm P(class)
            self.class_prior[c] = len(X_c) / total_samples

            # Tổng số lần xuất hiện của từng từ
            word_count = X_c.sum(axis=0)

            # Laplace smoothing
            self.feature_prob[c] = (
                word_count + self.alpha
            ) / (
                word_count.sum() + self.alpha * n_features
            )

    def predict(self, X_test):

        # Nếu X_test là sparse matrix thì chuyển sang ndarray
        if hasattr(X_test, "toarray"):
            X_test = X_test.toarray()

        predictions = []

        for x in X_test:

            scores = []

            for c in self.classes:

                # log(P(class))
                score = np.log(self.class_prior[c])

                # log(P(word|class))
                score += np.sum(
                    x * np.log(self.feature_prob[c])
                )

                scores.append(score)

            # Chọn lớp có score lớn nhất
            predictions.append(
                self.classes[np.argmax(scores)]
            )

        return np.array(predictions)

    def score(self, X_test, y_test):

        y_pred = self.predict(X_test)

        accuracy = np.mean(y_pred == y_test)

        return accuracy


def train_mnb_model(X_train_vec, y_train):

    print("Đang huấn luyện mô hình Multinomial Naive Bayes...")

    model = MyMultinomialNB(alpha=1.0)

    # Huấn luyện mô hình
    model.fit(X_train_vec, y_train)

    print("Huấn luyện hoàn tất!")

    return model

