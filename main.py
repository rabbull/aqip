import config
import time

import torch.nn as nn

from utils import meter


def main():
    if config.PROCESS == 'TRAIN':
        train()
    elif config.PROCESS == 'TEST':
        test()
    else:
        raise NotImplementedError


def train_epoch(model: nn.Module, criterion: nn.Module, data_loader, optimizer, site_id: int,
                log_freq=None):
    model.train()
    mse_meter = meter.MSEMeter(root=True)

    t0 = time.time()
    for i, data in enumerate(data_loader):
        seq, target = data['seq'], data['label']
        seq = seq.cuda().float()
        target = target[:, site_id]
        target = target.cuda().float()
        t1 = time.time()
        optimizer.zero_grad()
        aqi_pred = model(seq, site_id)
        target = target.view(-1)
        mse_meter.add(aqi_pred.detach(), target.detach())
        loss = criterion(aqi_pred, target)
        loss.backward()
        optimizer.step()
        t2 = time.time()
        if log_freq and (i + 1) % log_freq == 0:
            print(f"[{i}/{len(data_loader) - 1}] IterTime: {int((t2 - t0) * 1000)}\t"
                  f"DataTime: {int((t1 - t0) * 1000)}\t| "
                  f"ModelTime: {int((t2 - t1) * 1000)}\t| "
                  f"Loss: {loss.detach():.2f}\t| "
                  f"MSE: {mse_meter.value():.2f}")
        t0 = time.time()
    return mse_meter.value()


def train():
    pass


def test():
    pass


if __name__ == '__main__':
    main()
