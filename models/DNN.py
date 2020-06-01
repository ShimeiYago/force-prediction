from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers


def model1(input_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(200, input_dim=input_dim, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='linear'))      

    optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=True)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model


def model2(input_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(200, input_dim=input_dim, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='linear'))      

    optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=True)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model


def model3(input_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(300, input_dim=input_dim, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='linear'))      

    optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=True)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model


def model4(input_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(200, input_dim=input_dim, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='linear'))      

    optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=True)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model


def model5(input_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(200, input_dim=input_dim, activation='tanh'))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(30, activation='tanh'))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(3, activation='linear'))      

    optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=True)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model
