import numpy as np
from sklearn.model_selection import train_test_split

from preprocessing.load_data import load_eeg_data
from preprocessing.scale_data import scale_features
from models.cnn_model import build_cnn_model
from training.train_model import train_model
from evaluation.evaluate_model import evaluate_model
from utils.visualizations import plot_training_history
from utils.config import CONFIG

def main():
    X, y = load_eeg_data(CONFIG['data']['normal'], CONFIG['data']['diseased'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'])

    X_train, X_test = scale_features(X_train, X_test)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = build_cnn_model(input_shape=(X_train.shape[1], 1))
    model, history = train_model(model, X_train, y_train, X_test, y_test,
                                 epochs=CONFIG['model']['epochs'],
                                 batch_size=CONFIG['model']['batch_size'])

    evaluate_model(model, X_test, y_test)

    plot_training_history(history)

if __name__ == "__main__":
    main()
