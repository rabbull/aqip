import time
import torch

import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import config
import utils.meter as meter
from dataset import AirConditionDataset
from model.AQIP import AQIP


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
    training_data_set = AirConditionDataset('/mnt/airlab/data', seq_len=config.SEQ_LENGTH,
                                            pred_time_step=config.PRE_TIME_STEP, with_aqi=True)

    training_data_loader = DataLoader(training_data_set, shuffle=False, batch_size=config.BATCH_SIZE,
                                      num_workers=config.NUM_WORKERS)

    AQIP_net = AQIP(training_data_loader.dataset.adjacent_matrix, seq_len=config.SEQ_LENGTH, with_aqi=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(AQIP_net.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)

    # for epoch in config.MAX_EPOCH:
    #     train_epoch(AQIP_net, criterion, training_data_loader, optimizer,site_id=)



    pass


def test():
    pass


if __name__ == '__main__':
    main()
