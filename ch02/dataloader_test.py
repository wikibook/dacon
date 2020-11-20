from torch.utils.data import DataLoader
from src.utils import PandasDataset

# 배치 사이즈는 하이퍼파라미터로 사용자가 직접 정의할 수 있습니다. 
batch_size=32

# 학습 데이터 csv와 검증 데이터 csv 경로를 지정해 줍니다.
train_path = 'data/train_splited.csv'
val_path = 'data/val.csv'

# Loader를 통해 Batch 크기로 데이터를 반환합니다.
train_dataset = PandasDataset(train_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)

val_dataset = PandasDataset(val_path)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
