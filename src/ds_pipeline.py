from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
RANDOM_STATE = 1

def load_data():
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target
    return X, y

def split_data(X, y):
    return train_test_split(X,y, test_size=0.2, random_state=RANDOM_STATE)

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Train features:", X_train.shape)
    print("Test features:", X_test.shape)
    print("Train target:", y_train.shape)
    print("Test target:", y_test.shape)

main()