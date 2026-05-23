from src.train import train_model


X_train = [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
]

y_train = [0, 0, 1, 1]


X_test = [
    [5, 6]
]

y_test = [1]


model = train_model(
    X_train,
    y_train,
    X_test,
    y_test
)