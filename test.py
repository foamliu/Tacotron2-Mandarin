import time

import torch
from tqdm import tqdm

import config
from data_gen import TextMelLoader, collate_fn
from utils import parse_args

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model.eval()

    args = parse_args()

    valid_dataset = TextMelLoader(config.validation_files, config)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                               pin_memory=True, shuffle=False, num_workers=args.num_workers)

    num_samples = len(valid_dataset)
    print('num_samples: ' + str(num_samples))

    start = time.time()
    # Batches
    for batch in tqdm(valid_loader):
        model.zero_grad()
        x, y = model.parse_batch(batch)

        # Forward prop.
        y_pred = model(x)

    elapsed = time.time() - start

    print('{:.5f} seconds per sample'.format(elapsed / num_samples))
