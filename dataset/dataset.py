import csv
from datetime import datetime
import os

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from utils.AQI import cal_aqi


class AirConditionDataset(Dataset):
    def __init__(self, path: str, pred_time_step: int, dist_threshold: float = 5, seq_len: int = 8,
                 with_aqi: bool = True):
        self.__pred_time_step = pred_time_step
        self.__dist_threshold = dist_threshold
        self.__seq_len = seq_len

        sites_file_path = os.path.join(path, 'aq_sites_elv.csv')
        ac_file_path = os.path.join(path, 'aq_met_daily.csv')
        sites_file = open(sites_file_path, 'r')
        ac_file = open(ac_file_path, 'r')
        sites_reader = csv.reader(sites_file)
        ac_reader = csv.reader(ac_file)

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

        self.__n_sites = len(self.__site_positions)
        self.__site_positions = np.array(self.__site_positions, dtype='float32')
        self.__distances = np.zeros([self.__n_sites, self.__n_sites], dtype='float32')
        for i in range(self.__n_sites):
            self.__distances[i, :] = np.sqrt(np.sum((self.__site_positions - self.__site_positions[i, :]) ** 2, axis=1))
        self.__adj = self.__distances < np.sqrt(self.__dist_threshold)

        air_conditions = self.__air_conditions
        dim_data = len(air_conditions[min_date][0]) - 1
        self.__air_conditions = np.zeros([max_date - min_date + 1, self.__n_sites, dim_data + int(with_aqi)])
        for date in air_conditions.keys():
            air_condition = air_conditions[date]
            for ent in air_condition:
                self.__air_conditions[date - min_date, ent[0], :dim_data] = np.array(ent[1:])
            if with_aqi:
                self.__air_conditions[date - min_date, :, -1] = cal_aqi(self.__air_conditions[date - min_date, :, 2:8])

    def __len__(self):
        return self.__air_conditions.shape[0] - self.__seq_len - self.__pred_time_step + 1

    def __getitem__(self, idx):
        return {
            'seq': self.__air_conditions[idx:idx + self.__seq_len],
            'label': self.__air_conditions[idx + self.__seq_len + self.__pred_time_step]
        }

    @property
    def adjacent_matrix(self):
        return self.__adj


if __name__ == '__main__':
    dataset = AirConditionDataset('/mnt/airlab/data', seq_len=5, pred_time_step=7, with_aqi=True)
    data_loader = DataLoader(dataset, shuffle=True, batch_size=3)
    for idx, data in enumerate(data_loader):
        print(idx, data['seq'].shape, data['label'].shape)
        if idx > 2:
            break
    print(data_loader.dataset.adjacent_matrix.shape)
    # TODO: verify
