import os
import pandas as pd
import torch
from src.model import SkipConnectionModel
from src.utils import TestDataset
from torch.utils.data import DataLoader

# 모델 평가 시 GPU를 사용하기 위해서 설정.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 테스트 데이터 위치
path_test = 'data/test.csv'

# pth 파일들(모델 한개 예시)
# 학습을 통해 저장된 pth 파일들을 가져옵니다.
pth_bin = 'bin/test_43.pth' # 학습해서 이미 모델이 저장되어 있어야합니다.

# csv가 저장될 디렉토리를 미리 만들어 놓습니다.
if not os.path.exists('test'):  # 'test' 는 USER에 맞게 지정하시면 됩니다.
    os.mkdir('test')

########################################
######### 모델 하나에 대한 테스트 ##########
########################################

# Test Model
# 모델을 테스트하기 위해서 모델을 다시 정의합니다.
test_model = SkipConnectionModel(fn_in=226, fn_out=4)
teest_model = test_model.to(device)

# Test dataset을 불러옵니다.
test_data = TestDataset(path_test)
test_loader = DataLoader(test_data, batch_size=10000,  num_workers=0)

# 테스트 데이터를 불러와서 모델로 결과를 예측하고 그 결과를 파일로 씁니다.
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        outputs = test_model(data.float())
pred_test = outputs

sample_sub = pd.read_csv('data/sample_submission.csv', index_col=0)
layers = ['layer_1','layer_2','layer_3','layer_4']
submission = sample_sub.values + pred_test.cpu().numpy()

submission = pd.DataFrame(data=submission,columns=layers)
submission.to_csv('./test/submission.csv', index_label='id')


#######################################################################
### 아래 버전은 앙상블용 예제 코드입니다.
#######################################################################
# test 파일 경로
# path_test = 'data/test.csv'

# pth_list = os.listdir('bin')  # 'outputs' pth들이 저장된 경로
# print(pth_list) # pth 파일 리스트들을 확인합니다.

# # 모델들을 dictionary 형태로 정의하여 바로 사용할 수 있게 합니다.
# models = {
#     'model':TestModel(),
#     'model1': TestModel1(),
#     'model2': TestModel2(),
#     'model3': TestModel3(),
#     'model4': TestModel4(),
#     'model5': TestModel5(),
#     'model6': TestModel6()
# }


# 모델에 학습된 가중치를 올립니다.
# USER_BIN = 'bin/model.pth'
# weights = torch.load(USER_BIN, map_location='cuda:1')
# test_model.load_state_dict(weights)
# test_model = test_model.to(device)
# test_model.eval()


# 앙상블 할 모델에 대해서 파일을 씁니다.
# for pth in sorted(pth_list):
#     if pth[-3:] != 'pth':
#         pass
#     else:
#         if int(pth[0]) == 0:
#             test_model(pth, test_loader, model_type='model')
#         elif int(pth[0]) == 1:
#             test_model(pth, test_loader, model_type='model1')
#         elif int(pth[0]) == 2:
#             #test_model(pth, test_loader, model_type='model2')
#             pass
#         elif int(pth[0]) == 3:
#             test_model(pth, test_loader, model_type='model4')
#         elif int(pth[0]) > 3 and int(pth[0]) <7:
#             test_model(pth, test_loader, model_type='model5')
#         elif int(pth[0])>= 7:
#             test_model(pth, test_loader, model_type='model6')

# pth 가중치를 불러와서 모델을 테스트하고 그 결과 csv 파일을 씁니다.
# def test_model(path_pth, test_loader, model_type:str):
#     model = models[model_type]
#     ws = torch.load(f'./outputs/{path_pth}', map_location='cpu') # 불러옴
#     model.load_state_dict(ws)
#     model.eval()
    
#     with torch.no_grad(): 
#         for data in test_loader:
#             outputs = model(data.float())  # 모델을 테스트
#     pred_test = outputs

#     sample_sub = pd.read_csv('sample_submission.csv', index_col=0) 
#     layers = ['layer_1','layer_2','layer_3','layer_4']
#     submission = sample_sub.values + pred_test.numpy()

#     submission = pd.DataFrame(data=submission,columns=layers)
#     submission.to_csv(f'./test/{path_pth[:-4]}.csv', index_label='id') # test 경로에 csv 파일 저장

# def check_state(model):    
#     for val in model.state_dict().keys():
#         if val[-4:] =='bias':
#             pass
#         else:
#             print(f'{val} : {model.state_dict()[val].shape}')
