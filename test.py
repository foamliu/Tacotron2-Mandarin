import time

import numpy as np
import torch
from tqdm import tqdm

import config
from models.models import Tacotron2
from utils import text_to_sequence, HParams

if __name__ == '__main__':
    checkpoint = 'tacotron2-cn.pt'
    print('loading model: {}...'.format(checkpoint))
    model = Tacotron2(HParams())
    model.load_state_dict(torch.load(checkpoint))
    model = model.to('cpu')
    model.eval()

    filename = config.validation_files
    with open(filename, 'r') as file:
        lines = file.readlines()

    num_samples = len(lines)
    print('num_samples: ' + str(num_samples))

    elapsed = 0
    # Batches
    for line in tqdm(lines):
        tokens = line.strip().split('|')
        text = tokens[1]
        sequence = np.array(text_to_sequence(text))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()

        start = time.time()
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        end = time.time()
        elapsed = elapsed + (end - start)

    print('Elapsed: {:.5f} ms'.format(elapsed / num_samples * 1000))
