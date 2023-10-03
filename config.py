# Training hyperparameters
BATCH_SIZE = 16
save_model_path = "./SAVED_MODELS/best_main_GAGNet.pt",
step_show = 3760
n_epoch = 50
leanring_rate = 1
num_wokers = 4


# Dataset
SAMPLE_RATE = 16000
MAX_LENGTH = SAMPLE_RATE * 8
N_FFT = 400
HOP_LENGTH = 160
base_path = "../Data"


# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 16