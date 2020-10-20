import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter


class ModelRecorder:
    def __init__(self, save_file=None, optimizer: optim.Optimizer = None, comparator_dict=None,
                 summary_writer=None):
        self.save_file = save_file
        self.optimizer = optimizer
        self.summary_writer: SummaryWriter = summary_writer
        if comparator_dict is None:
            comparator_dict = {}
        self.best_epoch_dict = {}
        self.best_model_dict = {}
        self.comparator_dict = comparator_dict
        self.best_performance_dict = {}
        self.best_optimizer_dict = {}
        self.performance_record = []

    def load(self, ckpt_file: str, from_measurement: str):
        """
        Load model `state_dict` from ckpt_file
        :param ckpt_file: path to ckpt_file
        :param from_measurement: get the best model dict using the specified measurement
        :return: (Best model's state_dict, Epoch, Optimizer state_dict, Performance by 'from_measurement')
        """
        ckpt: dict = torch.load(ckpt_file)
        self.best_performance_dict = ckpt['best_performance_dict']
        self.best_model_dict = ckpt['best_model_dict']
        self.best_optimizer_dict = ckpt['best_optimizer_dict']
        self.best_epoch_dict = ckpt['best_epoch_dict']

        return self.best_model_dict[from_measurement], self.best_epoch_dict[from_measurement], self.best_optimizer_dict[
            from_measurement], self.best_performance_dict[from_measurement],

    def add(self, epoch, model: torch.nn.Module, performance_dict: dict, addition_info=None):
        self.performance_record.append(performance_dict)
        # record performance
        for key in performance_dict:
            # write to tensorboard
            if self.summary_writer is not None:
                self.summary_writer.add_scalar(str(key), performance_dict[key], global_step=epoch)
            # record the best
            curr_best = False
            if key not in self.best_performance_dict:
                curr_best = True
            else:
                is_better = self.__mse_comparator
                if key in self.comparator_dict:
                    is_better = self.comparator_dict[key]
                if is_better(performance_dict[key], self.best_performance_dict[key]):
                    curr_best = True
            if curr_best:
                # save as cpu tensor for faster preview (e.g. preview in Ipython)
                model_state_dict = self.__get_cpu_model_state_dict(model)
                self.best_performance_dict[key] = performance_dict[key]
                self.best_epoch_dict[key] = epoch
                self.best_model_dict[key] = model_state_dict
                self.best_optimizer_dict[key] = self.optimizer.state_dict()
        torch.save(dict(epoch=epoch,
                        best_epoch_dict=self.best_epoch_dict,
                        best_model_dict=self.best_model_dict,
                        best_performance_dict=self.best_performance_dict,
                        addition_info=addition_info,
                        best_optimizer_dict=self.best_optimizer_dict), self.save_file)

    @staticmethod
    def __default_comparator(x, y):
        # x is better than y
        return x > y

    @staticmethod
    def __mse_comparator(x, y):
        # smaller mse the better
        return x < y

    def print_best_stat(self):
        for key in self.best_performance_dict:
            print(f'best {key}: {self.best_performance_dict[key]} (at epoch {self.best_epoch_dict[key]})')

    def print_curr_stat(self, print_best=True):
        curr_performance_dict = self.performance_record[-1]
        for key in curr_performance_dict:
            print(f'curr {key}: {curr_performance_dict[key]} ')
        if print_best:
            self.print_best_stat()

    @staticmethod
    def __get_cpu_model_state_dict(model):
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        assert isinstance(model, torch.nn.Module)
        model_state_dict = model.state_dict()
        for k in model_state_dict:
            model_state_dict[k] = model_state_dict[k].cpu()
        return model_state_dict
