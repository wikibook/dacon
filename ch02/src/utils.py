import math
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

class PandasDataset(Dataset):
    """ Train dataset을 가져와서 torch 모델이 학습할 수 있는 tensor 형태로 반환합니다."""
    def __init__(self, path):
        super(PandasDataset, self).__init__()
        train = pd.read_csv(path).iloc[:,1:]
        self.train_X, self.train_Y = train.iloc[:,4:], train.iloc[:,0:4]
        self.tmp_x , self.tmp_y = self.train_X.values, self.train_Y.values
    
    def __len__(self):
        return len(self.train_X)

    def __getitem__(self, idx):
        return {
            'X':torch.from_numpy(self.tmp_x)[idx],
            'Y':torch.from_numpy(self.tmp_y)[idx]
        }

class TestDataset(Dataset):
    def __init__(self, path_test):
        super(TestDataset, self).__init__()
        test = pd.read_csv(path_test)
        self.test_X = test.iloc[:,1:]
        self.tmp_x = self.test_X.values
    
    def __len__(self):
        return len(self.test_X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.tmp_x)[idx]

"""
학습 최적화를 위해 스케줄러를 활용합니다.
Pytorch 및 transformer의 스케줄러를 참고.
https://github.com/huggingface/transformers/blob/master/src/transformers/optimization.py
"""
def get_constant_schedule(optimizer, last_epoch=-1):
    """ Create a schedule with a constant learning rate.
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """ Create a schedule with a constant learning rate preceded by a warmup
    period during which the learning rate increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=1.0, last_epoch=-1
):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function with several hard restarts, after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)