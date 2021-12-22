from options.options import Options


class ClassificationOptions(Options):
    def __init__(self):
        super().__init__()
        # dataset related
        self.batch_size_test = 256
        self.batch_size_train = 64

        # hyperparameters
        self.lr = 0.0001
        self.num_epochs = 20
        '''amount of nodes per layer'''
        self.hidden_sizes = [784, 64, 128, 64, 10]
