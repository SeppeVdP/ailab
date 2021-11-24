from options.options import Options

class LinearRegressionOptions(Options):
    def __init__(self):
        super().__init__()
        # dataset related
        self.batch_size_test = 7000
        self.batch_size_train = 7000
        self.train_dataset_size = 90000
        self.test_dataset_size = 60000
        self.min_house_size = 30
        self.max_house_size = 70
        self.noise_house_data = 50000

        # hyperparameters
        self.lr = 0.0001
        self.num_epochs = 20
