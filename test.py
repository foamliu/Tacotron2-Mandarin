import time

import torch
from tqdm import tqdm

import config
from data_gen import TextMelLoader, TextMelCollate
from models.models import Tacotron2
from utils import parse_args, HParams

if __name__ == '__main__':
    checkpoint = 'tacotron2-cn.pt'
    print('loading model: {}...'.format(checkpoint))
    model = Tacotron2(HParams())
    model.load_state_dict(torch.load(checkpoint))
    model = model.to('cpu')
    model.eval()

    args = parse_args()

    collate_fn = TextMelCollate(config.n_frames_per_step)

    valid_dataset = TextMelLoader(config.validation_files, config)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                               pin_memory=True, shuffle=False, num_workers=args.num_workers)

    num_samples = len(valid_dataset)
    print('num_samples: ' + str(num_samples))

    elapsed = 0
    # Batches
    for batch in tqdm(valid_loader):
        model.zero_grad()
        x, y = model.parse_batch(batch)

        # Forward prop.
        start = time.time()
        y_pred = model(x)
        end = time.time()
        elapsed = elapsed + (end - start)

    elapsed = time.time() - start

    print('Elapsed: {:.5f} ms'.format(elapsed / num_samples * 1000))
