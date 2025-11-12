import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pickle

class CIFAR10Dataset(Dataset):
    """데이터셋 구성 및 라벨링"""

    def __init__(self, images, labels, transform=None, phase='train'):
        self.images = images  # (N, 32, 32, 3)
        self.labels = labels
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]  # (32, 32, 3)
        label = self.labels[index]
        
        # 정규화: 0~255 → 0~1
        image = image.astype(np.float32) / 255.0
        
        # (H, W, C) → (C, H, W)
        image = image.transpose(2, 0, 1)
        
        # numpy → tensor
        image = torch.from_numpy(image)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image, self.phase)
            
        # label = torch.tensor(label, dtype=torch.long)

        return image, label

def load_cifar10_data(data_path, val_size=0.2, random_state=42):
    """
    CIFAR-10 데이터를 로드하고 train/val/test로 분할
    
    Args:
        data_dir: cifar-10-batches-py 폴더 경로
        val_size: 검증 데이터 비율
        random_state: 랜덤 시드
    
    Returns:
        x_train, y_train: 훈련 데이터
        x_val, y_val: 검증 데이터
        x_test, y_test: 테스트 데이터
    """
    # 훈련 데이터 로드 (5개 배치)
    x_train_list = []
    y_train_list = []
    
    for i in range(1, 6):
        file_path = os.path.join(data_path, f'data_batch_{i}')
        with open(file_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            x_train_list.append(batch[b'data'])
            y_train_list.append(batch[b'labels'])
    
    x_train_full = np.concatenate(x_train_list)  # (50000, 3072)
    y_train_full = np.concatenate(y_train_list)  # (50000,)
    
    # 테스트 데이터 로드
    test_file = os.path.join(data_path, 'test_batch')
    with open(test_file, 'rb') as f:
        test_batch = pickle.load(f, encoding='bytes')
        x_test = test_batch[b'data']  # (10000, 3072)
        y_test = np.array(test_batch[b'labels'])  # (10000,)
    
    # 데이터 reshape: (N, 3072) → (N, 3, 32, 32) → (N, 32, 32, 3)
    x_train_full = x_train_full.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    print(f"원본 훈련 데이터: {x_train_full.shape}")  # (50000, 32, 32, 3)
    print(f"원본 테스트 데이터: {x_test.shape}")     # (10000, 32, 32, 3)
    
    # 훈련 데이터를 train/validation으로 분할
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full,
        test_size=val_size,
        stratify=y_train_full,
        random_state=random_state
    )
    
    print(f"분할 후 훈련 데이터: {x_train.shape}")
    print(f"분할 후 검증 데이터: {x_val.shape}")
    
    return x_train, y_train, x_val, y_val, x_test, y_test