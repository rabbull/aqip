import time
import torch

import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import config
import utils.meter as meter
from dataset import AirConditionDataset
import dataset.dataset as ds
from model.AQIP import AQIP
from utils.ModelRecorder import ModelRecorder
from tensorboardX import SummaryWriter


def main():
    if config.PROCESS == 'TRAIN':
        train()
    elif config.PROCESS == 'TEST':
        test()
    else:
        raise NotImplementedError


def train_epoch(model: nn.Module, criterion: nn.Module, data_loader, optimizer, site_id: int, device,
                log_freq=None):
    model.train()
    mse_meter = meter.MSEMeter(root=True)
    t0 = time.time()
    for i, data in enumerate(data_loader):
        seq, target = data['seq'].to(device), data['label'].to(device)
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
    summary_writer = SummaryWriter(comment=config.COMMENT)
    training_data_set = AirConditionDataset(config.DATASET_DIR, seq_len=config.SEQ_LENGTH,
                                            pred_time_step=config.PRE_TIME_STEP, with_aqi=True, status=ds.STATUS_TRIAN)

    training_data_loader = DataLoader(training_data_set, shuffle=False, batch_size=config.BATCH_SIZE,
                                      num_workers=config.NUM_WORKERS)
    validation_data_set = AirConditionDataset(config.DATASET_DIR, seq_len=config.SEQ_LENGTH,
                                              pred_time_step=config.PRE_TIME_STEP, with_aqi=True,
                                              status=ds.STATUS_VALID)

    validation_data_loader = DataLoader(training_data_set, shuffle=False, batch_size=config.BATCH_SIZE,
                                        num_workers=config.NUM_WORKERS)

    print('count of train days: ', len(training_data_set))
    print('count of val days: ', len(validation_data_set))

    AQIP_net = AQIP(training_data_loader.dataset.adjacency_matrix, seq_len=config.SEQ_LENGTH, with_aqi=True)
    device = torch.device(config.CUDA_DEVICE)
    AQIP_net = AQIP_net.to(device)
    AQIP_net = nn.DataParallel(AQIP_net)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(config.CUDA_DEVICE)
    optimizer = torch.optim.SGD(AQIP_net.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)

    recorder = ModelRecorder(save_file=config.CKPT_FILE, optimizer=optimizer, summary_writer=summary_writer)

    resume_epoch = 0
    if config.RESUME:
        from_measurement = config.FROM_MEASUREMENT
        model_state_dict, saved_epoch, opt_state_dict, best_performance = recorder.load(config.CKPT_FILE,
                                                                                        from_measurement)
        print(f"Loading model from {config.CKPT_FILE},saved epoch: {saved_epoch},"
              f"{from_measurement}: {best_performance}")
        AQIP_net.module.load_state_dict(model_state_dict)
        optimizer.load_state_dict(opt_state_dict)
        resume_epoch = saved_epoch + 1
        print(f"Resume at epoch {resume_epoch}")

    # Train a model for every station ?
    # site_ids = range(0, training_data_loader.dataset.n_sites)

    for epoch in range(resume_epoch, config.MAX_EPOCH):
        optimizer.step()
        # train
        train_epoch(AQIP_net, criterion, training_data_loader, optimizer, config.SITE_ID, device)
        # validation
        with torch.no_grad():
            MSE = validation(validation_data_loader, AQIP_net, epoch)
        # save checkpoint
        recorder.add(epoch, AQIP_net, dict(MSE=MSE))
        recorder.print_curr_stat()
        print()

    summary_writer.close()
    print("\nTrain Finished!")
    pass


def test():
    pass


def validation(data_loader, net: AQIP, epoch, site_id: int):
    print(f'[ Validation summary ] epoch {epoch}:\n')
    rst = evaluate(data_loader, net, site_id)
    return rst


def evaluate(data_loader, net: nn.Module, site_id: int):
    mse_meter = meter.MSEMeter(root=True)
    net.eval()

    t0 = time.time()
    for i, data in enumerate(data_loader):
        seq, target = data['seq'], data['label']
        seq = seq.cuda().float()
        target = target[:, site_id]
        target = target.cuda().float()
        t1 = time.time()
        aqi_pred = net(seq, site_id)
        target = target.view(-1)
        mse_meter.add(aqi_pred.detach(), target.detach())
        t2 = time.time()
        if i % config.PRINT_FEQ == 0:
            print(f'[{i}/{len(data_loader)}]\t'
                  f'Batch Time {t1 - t0:.3f}\t'
                  f'Epoch Time {t2 - t1:.3f}\t'
                  f'acc(c) {mse_meter.value():.3f}')
    print(f'[ Validation summary ]:\n'
          f'MSE: {mse_meter.value()}')
    return mse_meter.value()


if __name__ == '__main__':
    main()
