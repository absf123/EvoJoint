import torch.nn as nn

def print_dict(dictionary, name=None):
    if name:
        print(f"{name}(")
    for key, value in dictionary.items():
        print(f"    {key}={value}")
    print(")")


class AttrDict(dict):
    def __init__(self, *config, **kwconfig):
        super(AttrDict, self).__init__(*config, **kwconfig)
        self.__dict__ = self
        for key in self:
            if type(self[key]) == dict:
                self[key] = AttrDict(self[key])

    def __getattr__(self, item):
        return None

    def get_values(self, keys):
        return {key: self.get(key) for key in keys}

    def dict(self):
        dictionary = dict(self)
        for key in dictionary:
            if type(dictionary[key]).__name__ == 'AttrDict':
                dictionary[key] = dict(dictionary[key])
        return dictionary


def initialize_weight(model, method):
    method = dict(normal=['normal_', dict(mean=0, std=0.01)],
                  xavier_uni=['xavier_uniform_', dict()],
                  xavier_normal=['xavier_normal_', dict()],
                  he_uni=['kaiming_uniform_', dict()],
                  he_normal=['kaiming_normal_', dict()]).get(method)
    if method is None:
        return None

    for module in model.modules():
        if module.__class__.__name__ in ['LSTM']:
            for param in module._all_weights[0]:
                if param.startswith('weight'):
                    getattr(nn.init, method[0])(getattr(module, param), **method[1])
                elif param.startswith('bias'):
                    nn.init.constant_(getattr(module, param), 0)
        else:
            if hasattr(module, "weight"):
                # Not BN
                if not ("BatchNorm" in module.__class__.__name__):
                    getattr(nn.init, method[0])(module.weight, **method[1])
                # BN
                else:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, "bias"):
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)


class Color:
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    PURPLE = '\033[95m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    @staticmethod
    def show():
        print(
            'Color(',
            f'    {Color.RED}RED{Color.END}',
            f'    {Color.YELLOW}YELLOW{Color.END}',
            f'    {Color.GREEN}GREEN{Color.END}',
            f'    {Color.BLUE}BLUE{Color.END}',
            f'    {Color.CYAN}CYAN{Color.END}',
            f'    {Color.DARKCYAN}DARKCYAN{Color.END}',
            f'    {Color.PURPLE}PURPLE{Color.END}',
            f'    {Color.BOLD}BOLD{Color.END}',
            f'    {Color.UNDERLINE}UNDERLINE{Color.END}',
            f'    {Color.END}END{Color.END}',
            ')',
            sep='\n'
        )
