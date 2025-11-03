import torch.nn as nn
import torch.nn.functional as F


# ======================================================
# 실험 1: 기본 모델
# ======================================================
class SimpleCNN(nn.Module):
    """
    CIFAR-10 분류를 위한 간단한 CNN 모델
    
    구조:
    - Conv1: 1채널 → 32채널, 3x3 필터
    - Conv2: 32채널 → 64채널, 3x3 필터
    - FC1: 평탄화 → 128 뉴런
    - FC2: 128 → 10 (클래스 개수)
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__() # nn.Module 초기화

        # 첫 번째 합성곱 레이어 (지역적 특징 추출)
        self.conv1 = nn.Conv2d(
            in_channels=3,    # 입력: 3채널
            out_channels=32,  # 출력: 32개 필터
            kernel_size=3,    # 3x3 필터
            stride=1,         # 이동 간격 1
            padding=1         # 패딩 1 (크기 유지)
        )
        self.bn1 = nn.BatchNorm2d(32)  # 배치 정규화 (학습 안정화, 수렴 속도 향상)
        self.relu1 = nn.ReLU()  # 활성화 함수(비선형성추가, 복잡한 패턴 학습 가능)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 맥스풀링 (크기 절반, 주요 특징 추출)
        
        # 두 번째 합성곱 레이어
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU() # ← 모듈로 정의됨
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 완전 연결 레이어
        # 입력 크기 계산: 32 → 16 (pool1) → 8 (pool2)
        # 64채널 × 8 × 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 128) # 완전 연결 레이어: 추출된 특징 요약, 분류기로 연결
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 드롭아웃 (과적합 방지)
        self.fc2 = nn.Linear(128, num_classes)  # 최종 출력 레이어

    def forward(self, x):
        """
        순전파 (Forward Pass)
            Args:
                x: 입력 이미지 (batch_size, 3, 32, 32)
            Returns:
                out: 클래스별 점수 (batch_size, 10)
        """
        # Conv1 블록
        out = self.conv1(x)       # (B, 1, 28, 28) → (B, 32, 28, 28)
        out = self.bn1(out)
        out = self.relu1(out)  # 모듈을 "호출" (클래스 인스턴스)
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
        out = self.relu3(out)
        out = self.dropout(out)   # 드롭아웃 적용 (훈련 시에만)
        out = self.fc2(out)       # (B, 128) → (B, 10)
        
        return out  # CrossEntropyLoss 사용 시 softmax 불필요


# ======================================================
# 실험 2-1: 활성화 함수 비교 (ReLU ↔ Sigmoid)
# ======================================================
class SimpleCNN_Sigmoid(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN_Sigmoid, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.Sigmoid()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.Sigmoid()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.act3 = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x):
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.act3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ======================================================
# 실험 2-2: 활성화 함수 비교 (ReLU ↔ Tanh)
# ======================================================
class SimpleCNN_Tanh(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN_Tanh, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.act3 = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.act3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ======================================================
# 실험 3: Dropout 비교 (0.0, 0.3, 0.5)
# ======================================================
class SimpleCNN_Dropout(nn.Module):
    def __init__(self, dropout_p=0.5, num_classes=10):
        super(SimpleCNN_Dropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ======================================================
# 실험 4: Overlapping Pooling (stride < kernel)
# ======================================================
class SimpleCNN_Overlapping(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN_Overlapping, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ======================================================
# 실험 6: BatchNorm ↔ LRN 비교
# ======================================================
class SimpleCNN_LRN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN_LRN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.lrn1 = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.lrn2 = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu1(self.lrn1(self.conv1(x))))
        x = self.pool(self.relu2(self.lrn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ============================================================================================================
  

def count_parameters(model):
    """
    학습 가능한 파라미터 개수 계산
        Returns:
            총 파라미터 개수
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) # requires_grad=True만 합산

def create_model(model_name="SimpleCNN", device="cuda", num_classes=10):
    """
    모델 생성 및 초기화
        Args:
            device: 'cuda' 또는 'cpu'
            num_classes: 클래스 개수
        Returns:
            model: 초기화된 모델
    """
    models = {
        "SimpleCNN": SimpleCNN,
        "Sigmoid": SimpleCNN_Sigmoid,
        "Tanh": SimpleCNN_Tanh,
        "Dropout": SimpleCNN_Dropout,
        "Overlapping": SimpleCNN_Overlapping,
        "LRN": SimpleCNN_LRN,
    }
    model = models[model_name](num_classes=num_classes).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")
    return model