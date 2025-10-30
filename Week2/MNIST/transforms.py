from torchvision import transforms

class MNISTTransform:
    """
    MNIST 이미지 전처리 클래스
    - train: 데이터 증강 적용
    - val: 변환 없이 정규화만 적용
    """
    
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
            transforms.RandomAffine(
                degrees=10,  # ±10도 랜덤 회전
                translate=(0.1, 0.1),  # 좌우/상하로 10% 이동
                scale=(0.9, 1.1)  # 90%~110% 크기 변환
            ),
            transforms.Normalize(mean, std)  # 정규화
        ])
        
        # 검증/테스트용 변환: 정규화만 적용
        self.val_transform = transforms.Compose([
            transforms.Normalize(mean, std)
        ])
    
    def __call__(self, image, phase='train'):
        """
        이미지에 변환 적용
        
        Args:
            image: 입력 이미지 텐서 (1, 28, 28)
            phase: 'train' 또는 'val'
        
        Returns:
            변환된 이미지 텐서
        """
        if phase == 'train':
            return self.train_transform(image)
        else:
            return self.val_transform(image)
