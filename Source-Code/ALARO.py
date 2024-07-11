import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class ALARO:
    def __init__(self, learning_rate=0.01):
        self.w_sim = 0.5
        self.w_edit = 0.5
        self.alpha = learning_rate
        self.expected_R = 0.0

        self.model = Sequential([
            Dense(64, input_dim=2, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def update_weights(self, R, sim, edit):
        self.w_sim += self.alpha * (R - self.expected_R) * sim
        self.w_edit -= self.alpha * (R - self.expected_R) * edit / max(sim, edit)
        return self.w_sim, self.w_edit

    def train_model(self, X, y, epochs=10):
        self.model.fit(X, y, epochs=epochs, verbose=1)
        return self.model

    def predict_reward(self, X):
        return self.model.predict(X)
