import os

import torch

from models.models import Tacotron2
from utils import HParams


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size: {}{}{} (MB)'.format(bcolors.OKBLUE, os.path.getsize("temp.p") / 1e6, bcolors.ENDC))
    os.remove('temp.p')


def test():
    checkpoint = 'tacotron2-cn.pt'
    print('loading model: {}...'.format(checkpoint))
    model = Tacotron2(HParams())
    model.load_state_dict(torch.load(checkpoint))
    model = model.to('cpu')
    model.eval()

    model.qconfig = torch.quantization.default_qconfig
    print(model.qconfig)
    torch.quantization.prepare(model, inplace=True)

    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)

    print("Size of model after quantization")
    print_size_of_model(model)


if __name__ == '__main__':
    test()
