#!/usr/bin/env python
# coding: utf-8

import os
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import config
from src.model import SkipConnectionModel
from src.utils import PandasDataset, get_cosine_with_hard_restarts_schedule_with_warmup

# 모델을 학습시키기 위한 하이퍼 파라미터를 설정합니다.
lr = config.LR
adam_epsilon = config.ADAM_EPSILON
epochs = config.EPOCHS
batch_size = config.BATCH_SIZE
warmup_step = config.WARMUP_STEPS

# original data : train.csv => random으로 섞어 미리 train_split data를 만들고
# evaluation을 위해서 val data를 따로 분리했습니다. (9:1)
# 모델 학습 데이터 경로를 설정합니다.
train_path = config.TRAIN_PATH
val_path = config.VAL_PATH

# Loader를 통해 Batch 데이터로 반환합니다.
train_dataset = PandasDataset(train_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)

val_dataset = PandasDataset(val_path)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0) 

# 모델이 학습하는 전체 step을 계산합니다.
total_step = len(train_loader) * epochs
print(f"Total step is....{total_step}")

# 모델 인스턴스를 정의합니다.
model = SkipConnectionModel(fn_in=226, fn_out=4)  # channel은 모델에서 수정합니다.

# GPU 및 CUDA 환경이 마련되어 있다면, 모델 학습을 위해 CUDA 환경을 직접 설정합니다.
# 그렇지 않은 경우 자동으로 CPU를 설정하게 됩니다.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델을 GPU 메모리에 올립니다. gpu가 없는 환경은 자동으로 cpu가 설정됩니다.
model = model.to(device) 

# 손실함수와 옵티마이저
# 신경망 모델을 최적화할 수 있게 손실함수를 정의합니다.
# MAE를 사용합니다.
loss_fn = nn.L1Loss()

# 옵티마이저와 스케줄러의 파라미터를 정의합니다.
no_decay = ["bias", "LayerNorm.weight"] # decay하지 않을 영역 지정
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() \
            if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {"params": [p for n, p in model.named_parameters() \
        if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]

# 옵티마이저와 스케줄러 객체를 정의합니다.
optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_step, num_training_steps=total_step
)

# 모델 이름을 위해서 변수를 만듭니다.
version = time.localtime()[3:5]
curr_lr = lr

# train loss와 val loss를 지정합니다.
total_loss = 0.0
total_val_loss = 0.0
n_val_loss = 10000000. # best validation loss를 저장하기 위해서 변수 설정합니다.

if not os.path.exists('bin'):
    os.mkdir('bin')

for epoch in range(epochs):
    total_loss = 0 
    total_val_loss = 0
    for i, data in enumerate(tqdm(train_loader, desc='*********Train mode*******')):  # train 데이터를 부르고 학습합니다.
        # 학습 데이터를 부르고 학습합니다.
        # 순방향 정의
        pred = model(data['X'].float().to(device))
        loss = loss_fn(pred, data['Y'].float().to(device))
        
        # 역방향 정의
        # optimizer 객체를 사용해서 학습 가능한 가중치 변수에 대한 모든 변화도를
        # 0으로 만듭니다.
        optimizer.zero_grad()

        # loss에 따른 오차 역전파를 구합니다.
        loss.backward()

        # optimizer 객체의 파라미터들을 업데이트합니다.
        optimizer.step()

        # scheduler 객체의 파라미터들을 업데이트합니다.
        scheduler.step()
        
        total_loss += loss.item()
        
    train_loss = total_loss / len(train_loader)
    print ("Epoch [{}/{}], Train Loss: {:.4f}".format(epoch+1, epochs, train_loss))

    # 평가
    # 검증 데이터를 부르고 에포크마다 학습된 모델을 부르고 평가합니다. 
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, \
            desc='*********Evaluation mode*******')):
            pred = model(data['X'].float().to(device))
            loss_val = loss_fn(pred, data['Y'].float().to(device))
            
            total_val_loss += loss_val.item()
    val_loss = total_val_loss / len(val_loader)
    print ("Epoch [{}/{}], Eval Loss: {:.4f}".format(epoch+1, epochs, val_loss))
    
    # 검증 데이터에서 가장 낮은 평균 절대 오차를 보인 에포크의 모델을 저장합니다.
    if val_loss < n_val_loss:
        n_val_loss = val_loss
        #torch.save(model.state_dict(), f'bin/test_{version}.pth')
        torch.save(model.state_dict(), f'bin/test.pth')
        print("Best Model saved......")
        