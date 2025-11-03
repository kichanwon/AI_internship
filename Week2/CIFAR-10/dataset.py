import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class MNISTDataset(Dataset):
    """데이터셋 구성 및 라벨링"""

    def __init__(self, images, labels, transform=None, phase='train'):
        """
        Args:
            images: numpy 배열 형태의 이미지 데이터 (N, 28, 28)
            labels: numpy 배열 형태의 레이블 (N,)
            transform: 이미지 변환 함수
            phase: 'train' 또는 'val' (데이터 증강 여부 결정)
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.phase = phase
    

    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.images)

    def __getitem__(self, index):
        """
        주어진 인덱스의 데이터를 반환
        
        Returns:
            image: 변환된 이미지 텐서 (1, 28, 28)
            label: 레이블 텐서
        """
        # 이미지와 레이블 가져오기
        image = self.images[index]  # (28, 28)
        label = self.labels[index]
        
        # 이미지 정규화: 0~255 범위를 0~1로 변환
        image = image.astype(np.float32) / 255.0
        
        # 채널 차원 추가: (28, 28) → (1, 28, 28)
        image = np.expand_dims(image, axis=0)
        
        # numpy 배열을 PyTorch 텐서로 변환
        image = torch.from_numpy(image)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image, self.phase)
            
        return image, label

def load_mnist_data(data_path, val_size=0.2, random_state=42):
    """
    MNIST 데이터를 로드하고 train/val/test로 분할
    
    Args:
        data_path: mnist.npz 파일 경로
        val_size: 검증 데이터 비율
        random_state: 랜덤 시드
    
    Returns:
        x_train, y_train: 훈련 데이터
        x_val, y_val: 검증 데이터
        x_test, y_test: 테스트 데이터
    """
    # npz 파일 로드
    data = np.load(data_path)
    
    # 훈련 데이터와 테스트 데이터 분리
    x_train_full = data['x_train']  # (60000, 28, 28)
    y_train_full = data['y_train']  # (60000,)
    x_test = data['x_test']         # (10000, 28, 28)
    y_test = data['y_test']         # (10000,)
    
    print(f"원본 훈련 데이터: {x_train_full.shape}")
    print(f"원본 테스트 데이터: {x_test.shape}")
    
    # 훈련 데이터를 train/validation으로 분할
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full,
        test_size=val_size,
        stratify=y_train_full,  # 클래스 비율 유지
        random_state=random_state
    )
    
    print(f"분할 후 훈련 데이터: {x_train.shape}")
    print(f"분할 후 검증 데이터: {x_val.shape}")
    
    return x_train, y_train, x_val, y_val, x_test, y_test