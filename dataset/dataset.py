import csv
from datetime import datetime
import os
import config
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from torch.utils.data import Dataset
import torch
from torch.utils.data.dataloader import DataLoader

from utils.AQI import cal_aqi

STATUS_TRAIN = 'train'
STATUS_VALID = 'validation'
STATUS_TEST = 'test'


class AirConditionDataset(Dataset):
    def __init__(self, path: str, pred_time_step: int, dist_threshold: float = 5, seq_len: int = 8,
                 with_aqi: bool = True, status=STATUS_TRAIN, adj_trainable=False):
        self.__pred_time_step = pred_time_step
        self.__dist_threshold = dist_threshold
        self.__seq_len = seq_len
        self.__with_aqi = with_aqi
        self.adj_trainable = adj_trainable
        sites_file_path = os.path.join(path, 'aq_sites_elv.csv')
        ac_file_path = os.path.join(path, 'aq_met_daily.csv')
        sites_file = open(sites_file_path, 'r')
        ac_file = open(ac_file_path, 'r')
        sites_reader = csv.reader(sites_file)
        ac_reader = csv.reader(ac_file)

        print(f"Dataset file read from: {sites_file_path} and {ac_file_path}")
        print(f"Data mode: {status}")

        # skip headers
        sites_reader.__next__()
        ac_reader.__next__()

        self.__sites_id_index_map = {}
        self.__site_positions = []
        for idx, row in enumerate(sites_reader):
            self.__site_positions.append([float(row[2]), float(row[3])])
            self.__sites_id_index_map[row[1]] = idx
        sites_file.close()

        self.__air_conditions = {}
        min_date = 1e9
        max_date = 0
        for row in ac_reader:
            site_index = self.__sites_id_index_map[row[0]]
            site_position = self.__site_positions[site_index]
            date = (datetime.strptime(row[1], "%Y-%m-%d") - datetime(2015, 1, 1)).days
            if date < min_date:
                min_date = date
            if date > max_date:
                max_date = date
            if date not in self.__air_conditions.keys():
                self.__air_conditions[date] = []
            self.__air_conditions[date].append(
                [site_index] + site_position + [float(d) if d != '' else 0 for d in row[2:]])
        ac_file.close()
        print("Dataset concatenated")

        self.n_sites = len(self.__site_positions)
        self.__site_positions = np.array(self.__site_positions, dtype='float32')
        self.__distances = np.zeros([self.n_sites, self.n_sites], dtype='float32')
        for i in range(self.n_sites):
            self.__distances[i, :] = np.sqrt(np.sum((self.__site_positions - self.__site_positions[i, :]) ** 2, axis=1))
        self.__adj = self.__distances < np.sqrt(self.__dist_threshold)


        ''' Use sparse representation of the matrix '''


        print(f"Adjacency matrix calculated for {self.n_sites} sites")

        air_conditions = self.__air_conditions
        dim_data = len(air_conditions[min_date][0]) - 1
        self.__air_conditions = np.zeros([max_date - min_date + 1, self.n_sites, dim_data + int(with_aqi)])
        for date in air_conditions.keys():
            air_condition = air_conditions[date]
            for ent in air_condition:
                self.__air_conditions[date - min_date, ent[0], :dim_data] = np.array(ent[1:])
        if with_aqi:
            self.__air_conditions[:, :, -1] = cal_aqi(self.__air_conditions[:, :, 2:8])
            print("AQI calculated")

        # Scale the dataset

        sizes = self.__air_conditions.shape
        scaler_seq = MinMaxScaler(feature_range=(0, 1))
        scaler_label = MinMaxScaler(feature_range=(0, 1))
        self.__air_conditions = self.__air_conditions.reshape(-1, self.__air_conditions.shape[2])
        self.__air_conditions[:, :-1] = scaler_seq.fit_transform(self.__air_conditions[:, :-1])
        self.__air_conditions[:, -1] = scaler_label.fit_transform(self.__air_conditions[:, -1].reshape(-1, 1)).reshape(-1)
        self.__data_scaler = scaler_label
        self.__air_conditions = self.__air_conditions.reshape(sizes[0], sizes[1], sizes[2])
        print("Dataset Scaled for LSTM")

        # Alter the dataset with respect to the status parameter
        if status is STATUS_TRAIN:
            self.__air_conditions = self.__air_conditions[:int(self.__air_conditions.shape[0] * 0.4), :, :]
        elif status is STATUS_TEST:
            self.__air_conditions = self.__air_conditions[
                                    int(self.__air_conditions.shape[0] * 0.4): int(self.__air_conditions.shape[0] * 0.7)
            , :, :]
        elif status is STATUS_VALID:
            self.__air_conditions = self.__air_conditions[
                                    int(self.__air_conditions.shape[0] * 0.7):, :, :]


        print(f"Final data length in days:{self.__len__()}")

    def __len__(self):
        # print("Get length method called")
        return self.__air_conditions.shape[0] - self.__seq_len - self.__pred_time_step

    def __getitem__(self, idx):
        # print(f"fetching data at index {idx} ")
        ret = {
            'seq': self.__air_conditions[idx:idx + self.__seq_len]
        }
        if self.__with_aqi:
            ret['label'] = self.__air_conditions[idx + self.__seq_len + self.__pred_time_step, :, -1]
        else:
            ret['label'] = cal_aqi(self.__air_conditions[idx + self.__seq_len + self.__pred_time_step, :, 2:8])
        return ret

    @property
    def adjacency_matrix(self):
        # The GConvLSTM need the adjacency matrix to be a trainable parameter
        if self.adj_trainable:
            return torch.tensor(self.__adj, dtype=torch.float)
        return self.__adj

    @property
    def scaler(self):
        return self.__data_scaler

if __name__ == '__main__':
    dataset = AirConditionDataset('/home/yuanhaozhu/GAT_covid19/data/', seq_len=config.SEQ_LENGTH,
                                  pred_time_step=config.PRE_TIME_STEP, with_aqi=True)
    # data_loader = DataLoader(dataset, shuffle=True, batch_size=16)
    # for idx, data in enumerate(data_loader):
    #     print(idx, data['seq'].shape, data['label'].shape)
    #     if idx > 2:
    #         break
    print(dataset.__getitem__(dataset.__len__() - 1))
    # TODO: verify
