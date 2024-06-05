from torch.utils.data import DataLoader
from . import transforms, folder_dataset
from utils.utils import Color


class DataGenerator:
    def __init__(self, config):
        self.config = config
        print(f"{Color.CYAN}[+] Data{Color.END}")
        self.train_loader = self.__make_data_loader(config, 'train')
        self.val_loader = self.__make_data_loader(config, 'val')
        self.test_loader = self.__make_data_loader(config, 'test')
        # Set input_shape and n_classes
        mini_batch_shape = list(self.train_loader.dataset.x.shape)
        mini_batch_shape[0] = None
        config.input_shape = mini_batch_shape
        config.n_classes = len(self.train_loader.dataset.y.unique())

        self.print_dataset()

    def __make_data_loader(self, config, phase):
        # Make dataset
        if config.folder:
            dataset = folder_dataset.FolderDataset(
                subject=config.subject,
                folder=config.folder,
                metric_learning=config.metric_learning,
                phase=phase)
        else:
            raise ValueError(f"Not supported datasets yet.")

        # Apply data transform
        compose = transforms.Compose(config.transform)  # 'ToTensor':True
        compose(dataset)

        # Return dataloader
        return DataLoader(dataset,
                          batch_size=config.batch_size,
                          shuffle=True if phase == 'train' else False,
                              drop_last=False)

    def print_dataset(self):
        if self.config.folder:
            print(f"Data folder: {self.config.folder}")
        else:
            print(f"Dataset: {self.config.dataset}")
        print(
            f"Subject: {self.config.subject}\n",
            f"Number of classes: {self.config.n_classes}",
            sep=''
        )
        print(f"Train set: {list(self.train_loader.dataset.x.shape)}")
        print(f"Validation set: {list(self.val_loader.dataset.x.shape)}")
        print(f"Test set: {list(self.test_loader.dataset.x.shape)}")
        print("")
