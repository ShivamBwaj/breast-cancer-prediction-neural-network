import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib


def build_and_train(path_model='model.h5', path_scaler='scaler.joblib'):
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_s.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_s, y_train, validation_split=0.1, epochs=20, batch_size=16)

    loss, acc = model.evaluate(X_test_s, y_test, verbose=0)
    print(f'Test accuracy: {acc:.4f}')

    model.save(path_model)
    joblib.dump(scaler, path_scaler)


if __name__ == '__main__':
    build_and_train()
