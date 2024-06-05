import argparse

class Config():
    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser(description='Two stage optimization')

        parser.add_argument("--batch_size", default=72, type=int, help="batch size")
        parser.add_argument('--training_epochs', default=500, type=int, help='training epochs')
        parser.add_argument('--perturbation_interval', default=100, type=int, help='perturbation interval')
        parser.add_argument('--num_iteration', default=2, type=int, help='number of loop iteration')
        parser.add_argument('--num_models', default=3, type=int, help='number of ensemble member')
        parser.add_argument('--num_samples', default=20, type=int, help='number of agents')
        parser.add_argument('--optim', default="RMSprop", type=str, help='optimizer')
        parser.add_argument("--momentum", default=0.9, type=float, help="optimizer momentum")

        # path
        parser.add_argument("--ray_model_path", default="ray_results", type=str, help="save path for ray model")
        parser.add_argument("--data_path", default="dataset", type=str, help="dataset path")  # recommend use the absolute path


        self.args = parser.parse_args()

        self.batch_size = self.args.batch_size
        self.training_epochs = self.args.training_epochs
        self.perturbation_interval = self.args.perturbation_interval
        self.num_iteration = self.args.num_iteration
        self.num_models = self.args.num_models
        self.num_samples = self.args.num_samples
        self.optim = self.args.optim
        self.momentum = self.args.momentum

        self.ray_model_path = self.args.ray_model_path
        self.data_path = self.args.data_path