import importlib
import torch

__all__ = ('Compose', 'ToTensor')

class Compose:
    def __init__(self, transforms):
        if transforms.__class__.__name__ not in ['AttrDict', 'dict']:
            raise TypeError(f"Not supported {transforms.__class__.__name__} type yet.")
        transforms_module = importlib.import_module('data_loader.transforms')
        configure_transform = lambda transform, params: getattr(transforms_module, transform)(**params) \
            if type(params).__name__ in ['dict', 'AttrDict'] else getattr(transforms_module, transform)(params)
        self.transforms = [configure_transform(transform, params) for transform, params in transforms.items()]

    def __call__(self, dataset):
        for transform in self.transforms:
            transform(dataset)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(\n'
        for transform in self.transforms:
            format_string += f'    {transform.__class__.__name__}()\n'
        format_string += ')'
        return format_string


class ToTensor:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, dataset):
        if type(dataset.x) != torch.Tensor:
            dataset.x = torch.as_tensor(dataset.x, dtype=torch.float)
        if type(dataset.y) != torch.Tensor:
            dataset.y = torch.as_tensor(dataset.y, dtype=torch.long)
