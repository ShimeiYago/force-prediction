class CycleLearningRate:
    def __init__(self, base_lr:float, max_lr:float, n_data:int, batchsize:int, mul=10):
        self.stepsize = (n_data // batchsize) * mul
        self.iteration = 0

        lr_list_of_1step = [base_lr + (max_lr-base_lr)/(self.stepsize-1) * i for i in range(self.stepsize)]
        self.cycle_list = lr_list_of_1step + lr_list_of_1step[1:-1][::-1]

    def __call__(self):
        i = self.iteration % (len(self.cycle_list))
        self.iteration += 1
        return self.cycle_list[i]

