PROCESS = 'TRAIN'  # should be either 'TRAIN' or 'TEST'
DATASET_DIR = '../data'  # path to dataset
RESULT_DIR = '../result'  # path to result
CUDA_DEVICE = 'cuda:0'  # gpu(s) to use
NUM_WORKERS = 2  # number of workers that loads data
BATCH_SIZE = 5
MAX_EPOCH = 300
RESUME = False
SEQ_LENGTH = 12
PRE_TIME_STEP = 1
LEARNING_RATE = 0.1
MOMENTUM = 0.8
KERNEL_SIZE = 3
COMMENT = f"LSTM"  # Should be one of the following: LSTM ,GAT+LSTM, GConvLSTM, GAT_CausalConv
RESULT_SUB_FOLDER = f"{RESULT_DIR}/{COMMENT}"
CKPT_FOLDER = f"{RESULT_SUB_FOLDER}/ckpt"
CKPT_FILE = f"{CKPT_FOLDER}/ckpt.pth"
CKPT_RECORD_FOLDER = f"{CKPT_FOLDER}/record"
FROM_MEASUREMENT = "MSE"  # The measurement taken to determine the best model
SITE_ID = 0  # The station ID that our model would be running on
PRINT_FEQ = 25  # The frequency parameter that used when printing the training information
DROP_OUT = 0.4  # The drop out probability parameter for the ST convolution network
'''
Best parameter for pure LSTM: LR=0.1 maxepoch=300, hidsize=128,batchsize=5 seqencelenth=12 and with two single layer 
                              LSTM module with tanh activation function manually operated to h,c and the output 
                              The optimizer is SGD 
'''