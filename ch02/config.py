#!/usr/bin/env python
# 경로와 학습에 필요한 configuration을 관리합니다. 리눅스 환경 기준으로 경로가 설정되어 있지만,
# 윈도우 환경에서 학습한다고 해도 파이썬 및 파이토치(cpu버전)이 설치되어 있다면 문제가 발생하지 않습니다.
TRAIN_PATH = 'data/train_splited.csv'
VAL_PATH = 'data/val.csv'

LR = 1e-03
ADAM_EPSILON = 1e-06
EPOCHS = 100
BATCH_SIZE = 2048
WARMUP_STEPS = 2000
