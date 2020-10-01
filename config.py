PROCESS = 'TRAIN'  # should be either 'TRAIN' or 'TEST'
DATASET_DIR = '../data'  # path to dataset
RESULT_DIR = '../result'  # path to result
DEVICES = '0,1,2,3'  # gpu(s) to use
NUM_WORKERS = 6  # number of workers that loads data
BATCH_SIZE = 16
MAX_EPOCH = 250
RESUME = False
