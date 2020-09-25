import csv
from datetime import datetime
import os

from torch.utils.data import Dataset


class AirConditionDataset(Dataset):
    def __init__(self, path: str, seq_size: int):
        sites_file_path = os.path.join(path, 'aq_sites_elv.csv')
        ac_file_path = os.path.join(path, 'aq_met_daily.csv')
        sites_file = open(sites_file_path, 'r')
        ac_file = open(ac_file_path, 'r')
        sites_reader = csv.reader(sites_file)
        ac_reader = csv.reader(ac_file)

        self.__sites = {}
        for row in sites_reader:
            self.__sites[row[1]] = [row[0], float(row[2]), float(row[3]), float(row[4])]

        self.__ac = {}
        for row in ac_reader:
            site_id = row[0]
            site = self.__sites[site_id]
            date = (datetime.strptime(row[1], '%Y-%m-%d') - datetime(2000, 1, 1)).days
            row = [float(col) if col != '' else 0 for col in row[2:]]
            if date not in self.__ac.keys():
                self.__ac[date] = []
            self.__ac[date].append(site[1:] + row[1:])

        sites_file.close()
        ac_file.close()

        self.__adjacent_matrix = {}
        for date in self.__ac.keys():
            records = self.__ac[date]
            print(records)
            # n = len(records)
            # self.__adjacent_matrix[date] = np.ones([n, n], dtype='float32')
            # for i, record in enumerate(records):
            #     self.__adjacent_matrix[i] = distance_based_influence(np.array(record[]), )
            input()


    def __len__(self):
        return len(self.__ac)

    def __getitem__(self, idx):
        pass

    def site(self, idx):
        return self.__sites[idx]

    def ac(self, idx):
        return self.__ac[idx]


if __name__ == '__main__':
    dataset = AirConditionDataset('/mnt/airlab/data', 0)
    for i in range(5):
        print(dataset[i])
