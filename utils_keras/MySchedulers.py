class LearningRate_StepDecay:
    def __init__(self, N_epochs, lr):
        self.N_epochs = N_epochs
        self.lr = lr

    def __call__(self, epoch):
        x = self.lr
        if epoch >= self.N_epochs // 2:
            x = x / 2
        if epoch >= (self.N_epochs // 4 * 3):
            x = x / 5
        return x


class LRtest:
    def __init__(self, lrlist):
        self.lrlist = lrlist

    def __call__(self, epoch):
        return self.lrlist[epoch]
