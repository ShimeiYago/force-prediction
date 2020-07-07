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
        elif model_number == 4:
            return self.model4()


    def model1(self):
        model = Sequential()
        model.add(Dense(400, input_dim=self.input_dim, activation='tanh'))
        model.add(Dense(200, activation='tanh'))
        model.add(Dense(100, activation='tanh'))
        model.add(Dense(50, activation='tanh'))
        model.add(Dense(30, activation='tanh'))
        model.add(Dense(10, activation='tanh'))
        model.add(Dense(3, activation='linear'))

        optimizer = optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=True)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        return model

    def model2(self):
        model = Sequential()
        model.add(Dense(800, input_dim=self.input_dim, activation='tanh'))
        model.add(Dense(600, activation='tanh'))
        model.add(Dense(400, activation='tanh'))
        model.add(Dense(200, activation='tanh'))
        model.add(Dense(100, activation='tanh'))
        model.add(Dense(100, activation='tanh'))
        model.add(Dense(100, activation='tanh'))
        model.add(Dense(50, activation='tanh'))
        model.add(Dense(30, activation='tanh'))
        model.add(Dense(10, activation='tanh'))
        model.add(Dense(3, activation='linear'))

        optimizer = optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=True)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        return model

    def model3(self):
        model = Sequential()
        model.add(Dense(800, input_dim=self.input_dim, activation='tanh'))
        model.add(Dense(600, activation='tanh'))
        model.add(Dense(400, activation='tanh'))
        model.add(Dense(200, activation='tanh'))
        model.add(Dense(100, activation='tanh'))
        model.add(Dense(100, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(50, activation='tanh'))
        model.add(Dense(30, activation='tanh'))
        model.add(Dense(10, activation='tanh'))
        model.add(Dense(3, activation='linear'))

        optimizer = optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=True)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        return model

    def model4(self):
        model = Sequential()
        model.add(Dense(400, input_dim=self.input_dim, activation='tanh'))
        model.add(Dense(100, activation='tanh'))
        model.add(Dense(50, activation='tanh'))
        model.add(Dense(10, activation='tanh'))
        model.add(Dense(3, activation='linear'))

        optimizer = optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=True)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        return model
