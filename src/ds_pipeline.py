
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore") 
RANDOM_STATE = 1

def load_data():
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target
    return X, y

def split_data(X, y):
    return train_test_split(X,y, test_size=0.2, random_state=RANDOM_STATE)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_model(X_train_scaled, y_train):
    mlp = MLPRegressor(random_state=RANDOM_STATE, hidden_layer_sizes=(10,5), max_iter=200, batch_size=1000,
                       activation="relu", validation_fraction=0.2, early_stopping=True)

    mlp.fit(X_train_scaled, y_train)
    return mlp

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    mlp = train_model(X_train_scaled, y_train)

    print("Train features:", X_train.shape)
    print("Test features:", X_test.shape)
    print("Train target:", y_train.shape)
    print("Test target:", y_test.shape)
    print("Model training complete.")
    print("Best validation score:", round(mlp.best_validation_score_, 3))

main()