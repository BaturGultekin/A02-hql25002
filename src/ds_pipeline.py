import matplotlib.pyplot as plt
import numpy as np
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
    mlp = MLPRegressor(random_state=RANDOM_STATE, hidden_layer_sizes=(16,8), max_iter=200, batch_size=128,
                       activation="relu", validation_fraction=0.2, early_stopping=True)

    mlp.fit(X_train_scaled, y_train)
    return mlp

def save_actual_vs_predicted_plot(y_true,y_pred, title, output_path):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))

    plt.plot([lo, hi], [lo, hi], linewidth=1, color="red")
    plt.xlabel("Actual MedHouseVal")
    plt.ylabel("Predicted MedHouseVal")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path, dpi=300)
    plt.close()

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

    y_pred_train = mlp.predict(X_train_scaled)
    y_pred_test = mlp.predict(X_test_scaled)
    save_actual_vs_predicted_plot(y_train, y_pred_train, "Actual vs Predicted MedHouseVal - Train", "figures/train_actual_vs_pred.png")
    save_actual_vs_predicted_plot(y_test, y_pred_test, "Actual vs Predicted MedHouseVal - Test", "figures/test_actual_vs_pred.png")

main()