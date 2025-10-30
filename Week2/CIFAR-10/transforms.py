from torchvision import transforms

class ImageTransform:
    """
    이미지 전처리를 위한 클래스
    데이터 증강(augmentation) → 일반화 향상 / 정확도 증가 가능
    """
    def __init__(self, resize, mean, std):
        """학습/검증용 변환 정의"""
        self.data_transform = {
            'train': transforms.Compose([ # 생성자: 출력 크기(resize), 정규화 mean/std를 인수로 받음
                transforms.Resize(256), # 짧은 변의 크기를 256으로 조정
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)), # 랜덤 영역을 크롭 후 리사이즈
                transforms.RandomHorizontalFlip(), # 50% 확률로 좌우 반전
                transforms.RandomRotation(45),  # +- N도 회전
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  # 밝기/대비/채도 랜덤 변경
                transforms.RandomGrayscale(p=0.1),  # 10% 확률로 흑백 변환 (RGB 채널 수 유지)
                transforms.ToTensor(), # PIL 또는 numpy.ndarray를 Tensor로 변환 / HWC -> CHW, 0-255를 0.0-1.0으로 스케일
                transforms.Normalize(mean, std) # 평균, 표준편차로 정규화
            ]),
            'val': transforms.Compose([ # 검증용(transform: 무작위성을 제거해 안정적 평가)
                transforms.Resize(256), # 짧은 변의 크기를 256으로 조정
                transforms.CenterCrop(resize), # 중앙 영역을 크롭 후 리사이즈
                transforms.ToTensor(), # 텐서 변환
                transforms.Normalize(mean, std) # 정규화
            ])
        }

    def __call__(self, img, phase):
        """입력 이미지에 변환 적용"""
        return self.data_transform[phase](img) # phase에 맞는 Compose 체인으로 img를 변환해 반환
