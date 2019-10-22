import torch

from config import device
from models.models import Tacotron2
from utils import HParams

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    # model.eval()

    torch.save(model.state_dict(), 'tacotron2-cn.pt')

    config = HParams()
    checkpoint = 'tacotron2-cn.pt'
    print('loading model: {}...'.format(checkpoint))
    model = Tacotron2(config)
    model.load_state_dict(torch.load(checkpoint))
    model = model.to(device)
    model.eval()
