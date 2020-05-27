from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers


class MyModel(Model):
    def __init__(self, input_dim, model_number):
        super(MyModel, self).__init__()
        if model_number == 0:
            self.layers_list = self._model0(input_dim)
    
    def __call__(self, x):
        for layer in self.layers_list:
            x = layer(x)
        return x

    def _model0(self, input_dim):
        middle_dims_list = [200, 100, 50, 30, 10]

        inlayer = Dense(middle_dims_list[0], input_dim=input_dim, activation='relu')
        midlayers = [Dense(dim, activation='relu') for dim in middle_dims_list[1:]]
        outlayer = Dense(3, activation='linear')

        return [inlayer] + midlayers + [outlayer]
