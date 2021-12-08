from options.options import Options


class ClassificationOptions(Options):
    def __init__(self):
        super().__init__()
        # dataset related
        self.batch_size_test = 1000
        self.batch_size_train = 16

        # hyperparameters
        self.lr = 0.000002
        self.num_epochs = 40
        '''amount of nodes per layer'''
        self.hidden_sizes = [784, 256, 32, 16, 10]
