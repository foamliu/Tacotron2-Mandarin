import os
import time

import numpy as np
import torch

import config
from models.models import Tacotron2
from utils import HParams, text_to_sequence


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def evaluate(model, neval_batches):
    model.eval()
    cnt = 0
    elapsed = 0
    filename = config.validation_files
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Batches
    for line in lines:
        tokens = line.strip().split('|')
        text = tokens[1]
        sequence = np.array(text_to_sequence(text))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()

        start = time.time()
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        end = time.time()
        elapsed = elapsed + (end - start)
        cnt += 1

        print('.', end='')
        if cnt >= neval_batches:
            break

    print('\nElapsed: {}{:.5f}{} sec'.format(bcolors.OKGREEN, elapsed, bcolors.ENDC))
    return


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

    print(bcolors.HEADER + '\nPost-training static quantization' + bcolors.ENDC)
    num_calibration_batches = 10

    model.qconfig = torch.quantization.default_qconfig
    print(model.qconfig)
    torch.quantization.prepare(model, inplace=True)

    # Calibrate first
    print('Post Training Quantization Prepare: Inserting Observers')

    # Calibrate with the training set
    print('Calibrate with the training set')
    evaluate(model, neval_batches=num_calibration_batches)
    print('Post Training Quantization: Calibration done')

    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    print('Post Training Quantization: Convert done')

    print("Size of model after quantization")
    print_size_of_model(model)


if __name__ == '__main__':
    test()
