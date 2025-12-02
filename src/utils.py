import os
import torch
import logging
from datetime import datetime
import numpy as np
import random


def get_device():
    # 跟随 CUDA_VISIBLE_DEVICES 自动选 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def setup_logger(log_dir='./logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pid = os.getpid()
    log_file = os.path.join(log_dir, f'run_{timestamp}_pid{pid}.log')

    logger = logging.getLogger(f'DisentangledCDM_{pid}')
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File Handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
