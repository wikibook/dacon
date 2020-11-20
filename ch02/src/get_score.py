#!/usr/bin/env python
# coding: utf-8

# ## 점수 복원 코드
# 
# - Baseline :  - MLP(ANN) 모델
#     - Skip Connection idea 적용 => 이전 정보를 효율적으로 활용
#     - LayerNorm : 블록마다 feature Normalization 사용하여 수렴을 촉진 
#     - GELU 활성화 함수 적용 (미분 가능 및 음수 값에 대한 계산 확대)
# 
# - 조합을 통해 대표로 3가지 구성 - 파생 6가지

import os
import math
import time
from itertools import chain

import numpy as np
import pandas as pd
import torch 
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


# ## Model define 
# >Test Model ~ Test Model 6
# 
# <code> weight list:
#     'test_(9, 49)_0.001_150.pth'-(819),'test_(20, 40)_0.001_80.pth-(2.92)', 'test_(20, 57)_0.001_100.pth'-(819),
#     'test_(12, 20)_0.001_100.pth'-(853),'test_(10, 7)_0.001_100.pth'-(867), 'test_(0, 54)_0.001_70.pth'-(819),
#     'test_(16, 47)_0.0001_80.pth'-(2.02),'test_(11, 43)_0.0001_70.pth-(2.02G)', 'test_(0, 9)_0.0001_70.pth'-(819),
#     'test_(19, 4)_0.001_200.pth'-(819)
#     
# </code>
# > 
# <code>
# TestModel : (12,20)
# TestModel1 : (0, 54)
# TestModel2 : (2.40)
# TestModel 4 : (10,7)
# TestModel 5 : (16,47), (11,43), (0,9)
# TestModel 6 : (19, 4), (20, 57), (9, 49)
# </code>

# In[2]:


# 각각의 모델들

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

        self.init_weights()

    def init_weights(self):
        self.weight.data.fill_(1.0)
        self.bias.data.zero_()

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

# Model 1 
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        
#         self.ln = LayerNorm(13000)
        self.ln = LayerNorm(10000)
        self.ln1 = LayerNorm(7000)
        self.ln2 = LayerNorm(4000)
        self.ln3 = LayerNorm(2000)
        
        self.upblock1 = nn.Sequential(nn.Linear(226, 2000),GELU(),nn.BatchNorm1d(2000))
        self.upblock2 = nn.Sequential(nn.Linear(2000,4000),GELU(),nn.BatchNorm1d(4000))
        self.upblock3 = nn.Sequential(nn.Linear(4000,7000), GELU(),nn.BatchNorm1d(7000))
        self.upblock4 = nn.Sequential(nn.Linear(7000,10000),GELU(),nn.BatchNorm1d(10000))
        #self.upblock5 = nn.Sequential(nn.Linear(10000,13000),GELU(),nn.BatchNorm1d(13000))

        #self.downblock1 = nn.Sequential(nn.Linear(13000, 10000),GELU(),nn.BatchNorm1d(10000))
        self.downblock1 = nn.Sequential(nn.Linear(10000, 7000),GELU(),nn.BatchNorm1d(7000))
        self.downblock2 = nn.Sequential(nn.Linear(7000, 4000),GELU(),nn.BatchNorm1d(4000))
        self.downblock3 = nn.Sequential(nn.Linear(4000, 2000),GELU(),nn.BatchNorm1d(2000))
        self.downblock4 = nn.Sequential(nn.Linear(2000, 300),GELU(),nn.BatchNorm1d(300))
        
        self.fclayer = nn.Sequential(nn.Linear(300,4))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        upblock1_out = self.upblock1(x)
        upblock2_out = self.upblock2(upblock1_out)
        upblock3_out = self.upblock3(upblock2_out)
        upblock4_out = self.upblock4(upblock3_out)
        #upblock5_out = self.upblock5(upblock4_out)
        
        downblock1_out = self.downblock1(self.ln(upblock4_out))
        skipblock1 = downblock1_out + upblock3_out
        downblock2_out = self.downblock2(self.ln1(skipblock1))
        skipblock2 = downblock2_out + upblock2_out
        downblock3_out = self.downblock3(self.ln2(skipblock2))
        skipblock3 = downblock3_out + upblock1_out
        downblock4_out = self.downblock4(self.ln3(skipblock3))
        
        output = self.fclayer(downblock4_out)
        
        return output
    
class TestModel1(nn.Module):
    def __init__(self):
        super(TestModel1, self).__init__()
        
#         self.ln = LayerNorm(13000)
        self.ln = LayerNorm(10000)
        self.ln1 = LayerNorm(7000)
        self.ln2 = LayerNorm(4000)
        self.ln3 = LayerNorm(1000)
        
        self.upblock1 = nn.Sequential(nn.Linear(226, 1000),GELU(),nn.BatchNorm1d(1000))
        self.upblock2 = nn.Sequential(nn.Linear(1000,4000),GELU(),nn.BatchNorm1d(4000))
        self.upblock3 = nn.Sequential(nn.Linear(4000,7000), GELU(),nn.BatchNorm1d(7000))
        self.upblock4 = nn.Sequential(nn.Linear(7000,10000),GELU(),nn.BatchNorm1d(10000))
        #self.upblock5 = nn.Sequential(nn.Linear(10000,13000),GELU(),nn.BatchNorm1d(13000))

        #self.downblock1 = nn.Sequential(nn.Linear(13000, 10000),GELU(),nn.BatchNorm1d(10000))
        self.downblock1 = nn.Sequential(nn.Linear(10000, 7000),GELU(),nn.BatchNorm1d(7000))
        self.downblock2 = nn.Sequential(nn.Linear(7000, 4000),GELU(),nn.BatchNorm1d(4000))
        self.downblock3 = nn.Sequential(nn.Linear(4000, 1000),GELU(),nn.BatchNorm1d(1000))
        self.downblock4 = nn.Sequential(nn.Linear(1000, 300),GELU(),nn.BatchNorm1d(300))
        
        self.fclayer = nn.Sequential(nn.Linear(300,4))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        upblock1_out = self.upblock1(x)
        upblock2_out = self.upblock2(upblock1_out)
        upblock3_out = self.upblock3(upblock2_out)
        upblock4_out = self.upblock4(upblock3_out)
        #upblock5_out = self.upblock5(upblock4_out)
        
        downblock1_out = self.downblock1(self.ln(upblock4_out))
        skipblock1 = downblock1_out + upblock3_out
        downblock2_out = self.downblock2(self.ln1(skipblock1))
        skipblock2 = downblock2_out + upblock2_out
        downblock3_out = self.downblock3(self.ln2(skipblock2))
        skipblock3 = downblock3_out + upblock1_out
        downblock4_out = self.downblock4(self.ln3(skipblock3))
        
        output = self.fclayer(downblock4_out)
        
        return output
    
# Model 2

class TestModel2(nn.Module):
    def __init__(self):
        super(TestModel2, self).__init__()
        
#         self.ln = LayerNorm(13000)
        self.ln = LayerNorm(20000)
        self.ln1 = LayerNorm(13000)
        self.ln2 = LayerNorm(7000)
        self.ln3 = LayerNorm(4000)
        self.ln4 = LayerNorm(1000)
        self.ln5 = LayerNorm(13000)
       
        
        self.upblock1 = nn.Sequential(nn.Linear(226, 1000),nn.ReLU(),nn.BatchNorm1d(1000))
        self.upblock2 = nn.Sequential(nn.Linear(1000,4000),nn.ReLU(),nn.BatchNorm1d(4000))
        self.upblock3 = nn.Sequential(nn.Linear(4000,7000),nn.ReLU(),nn.BatchNorm1d(7000))
        self.upblock4 = nn.Sequential(nn.Linear(7000,13000),nn.ReLU(),nn.BatchNorm1d(13000))
        self.upblock5 = nn.Sequential(nn.Linear(13000,20000),nn.ReLU(),nn.BatchNorm1d(20000))
        self.upblock6 = nn.Sequential(nn.Linear(20000,13000),nn.ReLU(),nn.BatchNorm1d(13000))

        self.downblock1 = nn.Sequential(nn.Linear(13000, 20000),nn.ReLU(),nn.BatchNorm1d(20000))
        self.downblock2 = nn.Sequential(nn.Linear(20000, 13000),nn.ReLU(),nn.BatchNorm1d(13000))
        self.downblock3 = nn.Sequential(nn.Linear(13000, 7000),nn.ReLU(),nn.BatchNorm1d(7000))
        self.downblock4 = nn.Sequential(nn.Linear(7000, 4000),nn.ReLU(),nn.BatchNorm1d(4000))
        self.downblock5 = nn.Sequential(nn.Linear(4000, 1000),nn.ReLU(),nn.BatchNorm1d(1000))
        self.downblock6 = nn.Sequential(nn.Linear(1000, 300),nn.ReLU(),nn.BatchNorm1d(300))
        
        
        self.fclayer = nn.Sequential(nn.Linear(300,4))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        upblock1_out = self.upblock1(x)
        upblock2_out = self.upblock2(upblock1_out)
        upblock3_out = self.upblock3(upblock2_out)
        upblock4_out = self.upblock4(upblock3_out)
        upblock5_out = self.upblock5(upblock4_out)
        upblock6_out = self.upblock6(upblock5_out)

        
        downblock1_out = self.downblock1(self.ln1(upblock6_out))
        skipblock1 = downblock1_out + upblock5_out  # 20000
        downblock2_out = self.downblock2(self.ln(skipblock1))
        skipblock2 = downblock2_out + upblock4_out # 13000
        downblock3_out = self.downblock3(self.ln5(skipblock2))
        skipblock3 = downblock3_out + upblock3_out # 7000
        downblock4_out = self.downblock4(self.ln2(skipblock3))
        skipblock4 = downblock4_out + upblock2_out # 4000
        
        downblock5_out = self.downblock5(self.ln3(skipblock4))
        skipblock5 = downblock5_out + upblock1_out
        downblock6_out = self.downblock6(self.ln4(skipblock5))
        
        output = self.fclayer(downblock6_out)
        
        return output

# Model3
class TestModel3(nn.Module):
    """
    Model for (20,40)
    """
    def __init__(self):
        super(TestModel3, self).__init__()
        
        self.ln = LayerNorm(17000)
        self.ln1 = LayerNorm(13000)
        self.ln2 = LayerNorm(7000)
        self.ln3 = LayerNorm(5000)        

        self.upblock1 = nn.Sequential(nn.Linear(226, 1000),nn.ReLU(),nn.BatchNorm1d(1000))
        self.upblock2 = nn.Sequential(nn.Linear(1000,3000),nn.ReLU(),nn.BatchNorm1d(3000))
        self.upblock3 = nn.Sequential(nn.Linear(3000,5000),nn.ReLU(),nn.BatchNorm1d(5000))
        self.upblock4 = nn.Sequential(nn.Linear(5000,7000),nn.ReLU(),nn.BatchNorm1d(7000))
        self.upblock5 = nn.Sequential(nn.Linear(7000,13000),nn.ReLU(),nn.BatchNorm1d(13000))
        self.upblock6 = nn.Sequential(nn.Linear(13000,17000),nn.ReLU(),nn.BatchNorm1d(17000))
        
        self.downblock1 = nn.Sequential(nn.Linear(17000,13000),nn.ReLU(),nn.BatchNorm1d(13000))
        self.downblock2 = nn.Sequential(nn.Linear(13000, 7000),nn.ReLU(),nn.BatchNorm1d(7000))
        self.downblock3 = nn.Sequential(nn.Linear(7000, 5000),nn.ReLU(),nn.BatchNorm1d(5000))
        self.downblock4 = nn.Sequential(nn.Linear(5000, 3000),nn.ReLU(),nn.BatchNorm1d(3000))
        self.downblock5 = nn.Sequential(nn.Linear(3000, 1000),nn.ReLU(),nn.BatchNorm1d(1000))
        self.downblock6 = nn.Sequential(nn.Linear(1000, 300),nn.ReLU(),nn.BatchNorm1d(300))
        
        self.fclayer = nn.Sequential(nn.Linear(300,4))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        upblock1_out = self.upblock1(x)
        upblock2_out = self.upblock2(upblock1_out)
        upblock3_out = self.upblock3(upblock2_out)
        upblock4_out = self.upblock4(upblock3_out)
        upblock5_out = self.upblock5(upblock4_out)
        upblock6_out = self.upblock6(upblock5_out)
                                    
        downblock1_out = self.dropout(self.downblock1(self.ln(upblock6_out)))
        skipblock1 = downblock1_out + upblock5_out
        downblock2_out = self.downblock2(self.ln1(skipblock1))
        skipblock2 = downblock2_out + upblock4_out
        downblock3_out = self.dropout(self.downblock3(self.ln2(skipblock2)))
        skipblock3 = downblock3_out + upblock3_out
        downblock4_out = self.downblock4(self.ln3(skipblock3))
        skipblock4 = downblock4_out + upblock2_out
        downblock5_out = self.downblock5(skipblock4)
        skipblock5 = self.dropout(downblock5_out + upblock1_out)
        downblock6_out = self.downblock6(skipblock5)
        
        output = self.fclayer(downblock6_out)
        
        return output

class TestModel4(nn.Module):
    def __init__(self):
        super(TestModel4, self).__init__()
        
        self.ln = LayerNorm(10000)
        self.ln1 = LayerNorm(7000)
        self.ln2 = LayerNorm(4000)
        self.ln3 = LayerNorm(1000)
        self.upblock1 = nn.Sequential(nn.Linear(226, 1000),nn.ReLU(),nn.BatchNorm1d(1000))
        self.upblock2 = nn.Sequential(nn.Linear(1000,10000),nn.ReLU(),nn.BatchNorm1d(10000))
        self.upblock3 = nn.Sequential(nn.Linear(10000,7000), nn.ReLU(),nn.BatchNorm1d(7000))
        self.upblock4 = nn.Sequential(nn.Linear(7000,4000),nn.ReLU(),nn.BatchNorm1d(4000))

        self.downblock1 = nn.Sequential(nn.Linear(4000, 7000),nn.ReLU(),nn.BatchNorm1d(7000))
        self.downblock2 = nn.Sequential(nn.Linear(7000, 10000),nn.ReLU(),nn.BatchNorm1d(10000))
        self.downblock3 = nn.Sequential(nn.Linear(10000, 1000),nn.ReLU(),nn.BatchNorm1d(1000))
        self.downblock4 = nn.Sequential(nn.Linear(1000, 300),nn.ReLU(),nn.BatchNorm1d(300))
        
        self.fclayer = nn.Sequential(nn.Linear(300,4))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        upblock1_out = self.upblock1(x)
        upblock2_out = self.dropout(self.upblock2(upblock1_out))
        upblock3_out = self.dropout(self.upblock3(upblock2_out))
        upblock4_out = self.dropout(self.upblock4(upblock3_out))
        
        downblock1_out = self.downblock1(self.ln2(upblock4_out))
        skipblock1 = downblock1_out + upblock3_out # 7000
        downblock2_out = self.downblock2(self.ln1(skipblock1))
        skipblock2 = downblock2_out + upblock2_out  # 10000
        downblock3_out = self.downblock3(self.ln(skipblock2))
        skipblock3 = downblock3_out + upblock1_out
        downblock4_out = self.downblock4(self.ln3(skipblock3))
        
        output = self.fclayer(downblock4_out)
        
        return output
    
class TestModel5(nn.Module):
    def __init__(self):
        super(TestModel5, self).__init__()
        
        self.ln = LayerNorm(13000)
        self.ln1 = LayerNorm(11000)
        self.ln2 = LayerNorm(7000)
        self.ln3 = LayerNorm(4000)
        self.upblock1 = nn.Sequential(nn.Linear(226, 1000),GELU(),nn.BatchNorm1d(1000))
        self.upblock2 = nn.Sequential(nn.Linear(1000,4000),GELU(),nn.BatchNorm1d(4000))
        self.upblock3 = nn.Sequential(nn.Linear(4000,7000), GELU(),nn.BatchNorm1d(7000))
        self.upblock4 = nn.Sequential(nn.Linear(7000,11000),GELU(),nn.BatchNorm1d(11000))
        self.upblock5 = nn.Sequential(nn.Linear(11000,13000),GELU(),nn.BatchNorm1d(13000))

        self.downblock1 = nn.Sequential(nn.Linear(13000, 11000),GELU(),nn.BatchNorm1d(11000))
        self.downblock2 = nn.Sequential(nn.Linear(11000, 7000),GELU(),nn.BatchNorm1d(7000))
        self.downblock3 = nn.Sequential(nn.Linear(7000, 4000),GELU(),nn.BatchNorm1d(4000))
        self.downblock4 = nn.Sequential(nn.Linear(4000, 1000),GELU(),nn.BatchNorm1d(1000))
        self.downblock5 = nn.Sequential(nn.Linear(1000, 300),GELU(),nn.BatchNorm1d(300))
        
        self.fclayer = nn.Sequential(nn.Linear(300,4))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        upblock1_out = self.upblock1(x)
        upblock2_out = self.upblock2(upblock1_out)
        upblock3_out = self.upblock3(upblock2_out)
        upblock4_out = self.upblock4(upblock3_out)
        upblock5_out = self.upblock5(upblock4_out)
        
        downblock1_out = self.downblock1(self.ln(upblock5_out))
        skipblock1 = downblock1_out + upblock4_out
        downblock2_out = self.downblock2(self.ln1(skipblock1))
        skipblock2 = downblock2_out + upblock3_out
        downblock3_out = self.downblock3(self.ln2(skipblock2))
        skipblock3 = downblock3_out + upblock2_out
        downblock4_out = self.dropout(self.downblock4(self.ln3(skipblock3)))
        skipblock4 = downblock4_out + upblock1_out
        downblock5_out = self.downblock5(skipblock4)
        
        output = self.fclayer(downblock5_out)
        
        return output

    
class TestModel6(nn.Module):
    def __init__(self):
        super(TestModel6, self).__init__()
        
#         self.ln = LayerNorm(13000)
        self.ln = LayerNorm(10000)
        self.ln1 = LayerNorm(7000)
        self.ln2 = LayerNorm(4000)
        
        self.upblock1 = nn.Sequential(nn.Linear(226, 1000),GELU(),nn.BatchNorm1d(1000))
        self.upblock2 = nn.Sequential(nn.Linear(1000,4000),GELU(),nn.BatchNorm1d(4000))
        self.upblock3 = nn.Sequential(nn.Linear(4000,7000), GELU(),nn.BatchNorm1d(7000))
        self.upblock4 = nn.Sequential(nn.Linear(7000,10000),GELU(),nn.BatchNorm1d(10000))
        #self.upblock5 = nn.Sequential(nn.Linear(10000,13000),GELU(),nn.BatchNorm1d(13000))

        #self.downblock1 = nn.Sequential(nn.Linear(13000, 10000),GELU(),nn.BatchNorm1d(10000))
        self.downblock1 = nn.Sequential(nn.Linear(10000, 7000),GELU(),nn.BatchNorm1d(7000))
        self.downblock2 = nn.Sequential(nn.Linear(7000, 4000),GELU(),nn.BatchNorm1d(4000))
        self.downblock3 = nn.Sequential(nn.Linear(4000, 1000),GELU(),nn.BatchNorm1d(1000))
        self.downblock4 = nn.Sequential(nn.Linear(1000, 300),GELU(),nn.BatchNorm1d(300))
        
        self.fclayer = nn.Sequential(nn.Linear(300,4))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        upblock1_out = self.upblock1(x)
        upblock2_out = self.upblock2(upblock1_out)
        upblock3_out = self.upblock3(upblock2_out)
        upblock4_out = self.upblock4(upblock3_out)
        #upblock5_out = self.upblock5(upblock4_out)
        
        downblock1_out = self.downblock1(self.ln(upblock4_out))
        skipblock1 = downblock1_out + upblock3_out
        downblock2_out = self.downblock2(self.ln1(skipblock1))
        skipblock2 = downblock2_out + upblock2_out
        downblock3_out = self.downblock3(self.ln2(skipblock2))
        skipblock3 = downblock3_out + upblock1_out
        downblock4_out = self.downblock4(skipblock3)
        
        output = self.fclayer(downblock4_out)
        
        return output


# ## Load weights and Test 
# 
# - train 된 pth 파일을 갖고 옵니다.
# - test file을 읽어서 evaluation mode로 모델을 테스트합니다.
# - 결과를 csv 파일로 반환합니다.
# 

# In[11]:


# 모델들을 dictionary 형태로 정의하여 바로 사용할 수 있게 합니다.
models = {
    'model':TestModel(),
    'model1': TestModel1(),
    'model2': TestModel2(),
    'model3': TestModel3(),
    'model4': TestModel4(),
    'model5': TestModel5(),
    'model6': TestModel6()
}


# 테스트 파일 경로
path_test = 'test.csv' 
# pth 파일 리스트들
pth_list = os.listdir('./outputs')  # 'outputs' pth들이 저장된 경로

print(pth_list)  # 2.pth > test_(10, 12)_0.0005_200 로 변경 예정

# csv가 저장될 디렉토리를 미리 만들어 놓습니다.
if os.path.exists('test'):  # 'test' 는 USER에 맞게 지정하시면 됩니다.
    pass
else: 
    os.mkdir('test')


# In[10]:


os.path.exists('test')


# In[4]:


# Test
# 테스트 데이터셋을 정의하고 부릅니다. 
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
    
test_data = TestDataset(path_test)
test_loader = DataLoader(test_data, batch_size=10000,  num_workers=4)


# In[5]:


# pth 가중치를 불러와서 모델을 테스트하고 그 결과 csv 파일을 씁니다.
def test_model(path_pth, test_loader, model_type:str):
    model = models[model_type]
    ws = torch.load(f'./outputs/{path_pth}', map_location='cpu') # 불러옴
    model.load_state_dict(ws)
    model.eval()
    
    with torch.no_grad(): 
        for data in test_loader:
            outputs = model(data.float())  # 모델을 테스트
    pred_test = outputs

    sample_sub = pd.read_csv('sample_submission.csv', index_col=0) 
    layers = ['layer_1','layer_2','layer_3','layer_4']
    submission = sample_sub.values + pred_test.numpy()

    submission = pd.DataFrame(data=submission,columns=layers)
    submission.to_csv(f'./test/{path_pth[:-4]}.csv', index_label='id') # test 경로에 csv 파일 저장


# In[6]:


# 앙상블 할 모델에 대해서 파일을 씁니다.
for pth in sorted(pth_list):
    if pth[-3:] != 'pth':
        pass
    else:
        if int(pth[0]) == 0:
            test_model(pth, test_loader, model_type='model')
        elif int(pth[0]) == 1:
            test_model(pth, test_loader, model_type='model1')
        elif int(pth[0]) == 2:
            #test_model(pth, test_loader, model_type='model2')
            pass
        elif int(pth[0]) == 3:
            test_model(pth, test_loader, model_type='model4')
        elif int(pth[0]) > 3 and int(pth[0]) <7:
            test_model(pth, test_loader, model_type='model5')
        elif int(pth[0])>= 7:
            test_model(pth, test_loader, model_type='model6')


# In[7]:


def check_state(model):    
    for val in model.state_dict().keys():
        if val[-4:] =='bias':
            pass
        else:
            print(f'{val} : {model.state_dict()[val].shape}')
