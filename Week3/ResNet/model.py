import torch.nn as nn
import torch.nn.functional as F
import torch
import config









class BasicBlock(nn.Module):
    """
    ResNet의 기본 빌딩 블록
    
    구조:
        x → Conv → BN → Act → Conv → BN → (+) → Act
        |___________________________________|
                  (skip connection)
    """
    expansion = 1  # 출력 채널 확장 비율
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, activation='relu'):
        super(BasicBlock, self).__init__()
        
        # 활성화 함수 선택
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 첫 번째 Conv: stride를 통해 다운샘플링 가능
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 두 번째 Conv: 항상 stride=1
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsample: 입력과 출력의 차원이 다를 때 사용 (skip connection 조정)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x  # Skip connection을 위한 입력 저장
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # Residual 연결
        out = self.activation(out)
        
        return out









# ======================================================
# ResNet Bottleneck Block (ResNet-50, ResNet-101, ResNet-152용)
# ======================================================
class BottleneckBlock(nn.Module):
    """
    ResNet의 Bottleneck 블록 (더 깊은 네트워크용)
    
    구조:
        x → 1x1 Conv → BN → Act → 3x3 Conv → BN → Act → 1x1 Conv → BN → (+) → Act
        |__________________________________________________________________|
                              (skip connection)
    
    1x1 Conv로 채널을 줄였다가 다시 늘려서 계산 효율성 증가
    """
    expansion = 4  # 출력 채널 = out_channels * 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, activation='relu'):
        super(BottleneckBlock, self).__init__()
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 1x1 Conv: 채널 축소
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 Conv: 공간적 특징 추출
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 Conv: 채널 확장 (out_channels * 4)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        # Bottleneck path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.activation(out)
        
        return out






# ======================================================
# ResNet 메인 아키텍처
# ======================================================
class ResNet(nn.Module):
    """
    ResNet 구현 (CIFAR-10용으로 수정)
    
    원본 ImageNet용 ResNet과의 차이점:
    - 첫 Conv: 7x7 stride=2 → 3x3 stride=1 (CIFAR-10은 32x32로 작음)
    - MaxPool 제거 (공간 차원 보존)
    """
    
    def __init__(self, block, layers, num_classes=10, activation='relu'):
        """
        Args:
            block: BasicBlock 또는 BottleneckBlock
            layers: 각 stage의 블록 개수 [stage1, stage2, stage3, stage4]
                   예) [2, 2, 2, 2] = ResNet-18
            num_classes: 출력 클래스 수
            activation: 활성화 함수 종류
        """
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.activation_fn = activation
        
        # 초기 Conv 레이어 (CIFAR-10용으로 수정)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # ResNet Stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global Average Pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        ResNet Stage 생성
        
        Args:
            block: BasicBlock 또는 BottleneckBlock
            out_channels: 출력 채널 수
            blocks: 블록 반복 횟수
            stride: 첫 블록의 stride (다운샘플링용)
        """
        downsample = None
        
        # 입력/출력 차원이 다르면 downsample 필요
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        # 첫 블록 (stride 적용, downsample 가능)
        layers.append(block(self.in_channels, out_channels, stride, downsample, self.activation_fn))
        
        # 나머지 블록들 (stride=1, downsample 없음)
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, activation=self.activation_fn))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Kaiming 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 초기 Conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        # ResNet Stages
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.layer4(x)  # 4x4
        
        # Global Average Pooling
        x = self.avgpool(x)  # 1x1
        x = torch.flatten(x, 1)
        
        # Fully Connected
        x = self.fc(x)
        
        return x





# ======================================================
# 사전 정의된 ResNet 모델들
# ======================================================
def ResNet18(num_classes=10, activation='relu'):
    """ResNet-18: [2, 2, 2, 2]"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, activation)

def ResNet34(num_classes=10, activation='relu'):
    """ResNet-34: [3, 4, 6, 3]"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, activation)

def ResNet50(num_classes=10, activation='relu'):
    """ResNet-50: [3, 4, 6, 3] with Bottleneck"""
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes, activation)

def ResNet101(num_classes=10, activation='relu'):
    """ResNet-101: [3, 4, 23, 3] with Bottleneck"""
    return ResNet(BottleneckBlock, [3, 4, 23, 3], num_classes, activation)

def ResNet152(num_classes=10, activation='relu'):
    """ResNet-152: [3, 8, 36, 3] with Bottleneck"""
    return ResNet(BottleneckBlock, [3, 8, 36, 3], num_classes, activation)







# ======================================================
# 모델 생성 함수 (기존 create_model 대체용)
# ======================================================
def count_parameters(model):
    """학습 가능한 파라미터 개수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_model(model_name="ResNet18", device="cuda", num_classes=10):
    """
    모델 생성 및 초기화
    
    Args:
        model_name: 'ImprovedCNN', 'ResNet18', 'ResNet34', 'ResNet50' 등
        device: 'cuda' 또는 'cpu'
        num_classes: 클래스 개수
    
    Returns:
        model: 초기화된 모델
    """
    activation = config.ACTIVATION_FUNCTION
    
    models = {
        "ImprovedCNN": lambda: ImprovedCNN(num_classes, activation),
        "ResNet18": lambda: ResNet18(num_classes, activation),
        "ResNet34": lambda: ResNet34(num_classes, activation),
        "ResNet50": lambda: ResNet50(num_classes, activation),
        "ResNet101": lambda: ResNet101(num_classes, activation),
        "ResNet152": lambda: ResNet152(num_classes, activation),
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    model = models[model_name]().to(device)
    print(f"Model: {model_name}")
    print(f"Activation: {activation}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    return model


# ImprovedCNN은 기존 코드 유지
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10, activationFn='tanh'):
        super(ImprovedCNN, self).__init__()
        activationFn = config.ACTIVATION_FUNCTION
        if activationFn == 'relu':
            self.activation = nn.ReLU()
        elif activationFn == 'tanh':
            self.activation = nn.Tanh()
        elif activationFn == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation function")

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