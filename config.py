import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 365
TOTAL_STEPS = 6000
LR = .001
ALPHA = 1
BETA = .01
