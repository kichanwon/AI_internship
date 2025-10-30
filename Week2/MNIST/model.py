import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    MNIST 분류를 위한 간단한 CNN 모델
    
    구조:
    - Conv1: 1채널 → 32채널, 3x3 필터
    - Conv2: 32채널 → 64채널, 3x3 필터
    - FC1: 평탄화 → 128 뉴런
    - FC2: 128 → 10 (클래스 개수)
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__() # nn.Module 초기화

        # 첫 번째 합성곱 레이어
        self.conv1 = nn.Conv2d(
            in_channels=1,    # 입력: 흑백 이미지 (1채널)
            out_channels=32,  # 출력: 32개 필터
            kernel_size=3,    # 3x3 필터
            stride=1,         # 이동 간격 1
            padding=1         # 패딩 1 (크기 유지)
        )
        self.bn1 = nn.BatchNorm2d(32)  # 배치 정규화 (학습 안정화)
        self.relu1 = nn.ReLU()  # 활성화 함수
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 맥스풀링 (크기 절반)
        
        # 두 번째 합성곱 레이어
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 완전 연결 레이어
        # 입력 크기 계산: 28 → 14 (pool1) → 7 (pool2)
        # 64채널 × 7 × 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)  # 드롭아웃 (과적합 방지)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        순전파 (Forward Pass)
            Args:
                x: 입력 이미지 (batch_size, 1, 28, 28)
            Returns:
                out: 클래스별 점수 (batch_size, 10)
        """
        # Conv1 블록
        out = self.conv1(x)       # (B, 1, 28, 28) → (B, 32, 28, 28)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)     # (B, 32, 28, 28) → (B, 32, 14, 14)
        
        # Conv2 블록
        out = self.conv2(out)     # (B, 32, 14, 14) → (B, 64, 14, 14)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)     # (B, 64, 14, 14) → (B, 64, 7, 7)
        
        # 평탄화 (Flatten)
        out = out.view(out.size(0), -1)  # (B, 64, 7, 7) → (B, 3136)
        
        # FC 레이어
        out = self.fc1(out)       # (B, 3136) → (B, 128)
        out = F.relu(out)
        out = self.dropout(out)   # 드롭아웃 적용 (훈련 시에만)
        out = self.fc2(out)       # (B, 128) → (B, 10)
        
        return out  # CrossEntropyLoss 사용 시 softmax 불필요

def count_parameters(model):
    """
    학습 가능한 파라미터 개수 계산
        Returns:
            총 파라미터 개수
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) # requires_grad=True만 합산

def create_model(device, num_classes=10):
    """
    모델 생성 및 초기화
        Args:
            device: 'cuda' 또는 'cpu'
            num_classes: 클래스 개수
        Returns:
            model: 초기화된 모델
    """
    model = SimpleCNN(num_classes=num_classes).to(device)
    print(f'Trainable parameters: {count_parameters(model):,}')
    return model
