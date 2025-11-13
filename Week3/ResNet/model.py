import torch.nn as nn
import torch.nn.functional as F
import config

# ======================================================
# 실험 3: 활성화 함수 레이어 추가 비교 (nn.Sequential)
# ======================================================
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10,activationFn='tanh'):
        super(ImprovedCNN, self).__init__()
        activationFn=config.ACTIVATION_FUNCTION
        if activationFn=='relu':
            self.activation = nn.ReLU()
        elif activationFn=='tanh':
            self.activation = nn.Tanh()
        elif activationFn=='sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation function")

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            self.activation,

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            self.activation,
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            self.activation,

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            self.activation,
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            self.activation,

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            self.activation,
            nn.MaxPool2d(2, 2)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

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
        "ImprovedCNN": ImprovedCNN
    }
    model = models[model_name](num_classes=num_classes).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")
    return model