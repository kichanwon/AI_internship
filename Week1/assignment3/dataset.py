import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split

class DogvsCatDataset(Dataset):
    """데이터셋 구성 및 라벨링"""
    
    def __init__(self, file_list, transform=None, phase='train'):
        """파일 목록과 라벨 생성, 변환기 저장"""
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        
        self.labels = []
        for path in self.file_list:
            folder = os.path.basename(os.path.dirname(path))
            label = 1 if folder == "dog" else 0
            self.labels.append(label)

    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.file_list)
    
    def __getitem__(self, index):
        """이미지 불러오기 + 변환 + 라벨 반환"""
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img, self.phase)
        
        label = torch.tensor(self.labels[index], dtype=torch.long)
        
        return img_transformed, label

def prepare_data(cat_directory, dog_directory, test_size=0.1, val_size=0.2, random_state=42):
    """폴더에서 이미지 수집 + 학습/검증/테스트 세트로 분할"""

    # 이미지 파일 경로 수집
    cat_images = sorted([os.path.join(cat_directory, f) for f in os.listdir(cat_directory)])
    dog_images = sorted([os.path.join(dog_directory, f) for f in os.listdir(dog_directory)])
    
    all_images = cat_images + dog_images
    labels = [0] * len(cat_images) + [1] * len(dog_images)
    
    # 훈련+검증 / 테스트 분할
    trainval_files, test_files, trainval_labels, test_labels = train_test_split(
        all_images, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    
    # 훈련 / 검증 분할
    train_files, val_files, train_labels, val_labels = train_test_split(
        trainval_files, trainval_labels, test_size=val_size, 
        stratify=trainval_labels, random_state=random_state
    )
    
    return train_files, val_files, test_files