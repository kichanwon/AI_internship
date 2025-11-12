from torchvision import transforms
import torch

class CIFARTransform:
    """CIFAR-10 이미지 전처리 클래스"""

    def __init__(self, mean, std):
        """
        Args:
            mean: 정규화에 사용할 평균값
            std: 정규화에 사용할 표준편차
        """
        self.mean = mean
        self.std = std
        
        # 훈련용 변환: 데이터 증강 포함
        self.train_transform = transforms.Compose([
            # transforms.RandomAffine(
            #     degrees=10,  # ±10도 랜덤 회전
            #     translate=(0.1, 0.1),  # 좌우/상하로 10% 이동
            #     scale=(0.9, 1.1)  # 90%~110% 크기 변환
            # ),
            transforms.RandomHorizontalFlip(p=0.5),  # 좌우 반전
            transforms.RandomCrop(32, padding=4),    # 랜덤 크롭
            # transforms.ColorJitter(                # 색상 변화
            #     brightness=0.2,
            #     contrast=0.2,
            #     saturation=0.2,
            #     hue=0.1
            # ),
            # transforms.ToTensor(),
            transforms.Normalize(mean, std)  # 정규화
        ])
        # 검증/테스트용 변환: 정규화만 적용
        self.val_transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        

    
    def __call__(self, image, phase='train'):
        if phase == 'train':
            return self.train_transform(image)
        else:
            return self.val_transform(image)
