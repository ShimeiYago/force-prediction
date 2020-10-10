from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers


class DNN:
    def __init__(self, input_dim, learning_rate=0.001):
        self.input_dim = input_dim
        self.learning_rate = learning_rate

    def __call__(self, model_number):
        if model_number == 1:
            return self.model1()
        elif model_number == 2:
            return self.model2()
        elif model_number == 3:
            return self.model3()

    def model1(self):
        model = Sequential()
        model.add(Dense(1024, input_dim=self.input_dim, activation='tanh'))
        model.add(Dense(512, activation='tanh'))
        model.add(Dense(256, activation='tanh'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(64, activation='tanh'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(16, activation='tanh'))
        model.add(Dense(3, activation='linear'))

        optimizer = optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=True)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        return model

    def model2(self):
        model = Sequential()
        model.add(Dense(1024, input_dim=self.input_dim, activation='tanh'))
        model.add(Dense(512, activation='tanh'))
        model.add(Dense(512, activation='tanh'))
        model.add(Dense(256, activation='tanh'))
        model.add(Dense(256, activation='tanh'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(64, activation='tanh'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(16, activation='tanh'))
        model.add(Dense(3, activation='linear'))

        optimizer = optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=True)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        return model
