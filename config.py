PROCESS = 'TRAIN'  # should be either 'TRAIN' or 'TEST'
DATASET_DIR = '../data'  # path to dataset
RESULT_DIR = '../result'  # path to result
CUDA_DEVICE = 'cuda:0'  # gpu(s) to use
NUM_WORKERS = 2  # number of workers that loads data
BATCH_SIZE = 3
MAX_EPOCH = 150
RESUME = False
SEQ_LENGTH = 10
PRE_TIME_STEP = 1
LEARNING_RATE = 0.01
MOMENTUM = 0.8
KERNEL_SIZE = 3
COMMENT = f"GAT+LSTM"
RESULT_SUB_FOLDER = f"{RESULT_DIR}/{COMMENT}"
CKPT_FOLDER = f"{RESULT_SUB_FOLDER}/ckpt"
CKPT_FILE = f"{CKPT_FOLDER}/ckpt.pth"
CKPT_RECORD_FOLDER = f"{CKPT_FOLDER}/record"
FROM_MEASUREMENT = "MSE"  # The measurement taken to determine the best model
SITE_ID = 23
PRINT_FEQ = 25
DROP_OUT = 0.4
'''
Best parameter for pure LSTM: LR=0.1 maxepoch=300, hidsize=128,batchsize=5 seqencelenth=12 and with two single layer 
                              LSTM module with tanh activation function manually operated to h,c and the output 
'''