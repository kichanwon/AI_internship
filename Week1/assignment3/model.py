import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """LeNet 기반의 CNN 모델"""
    
    def __init__(self, num_classes=2):
        super(LeNet, self).__init__()  # nn.Module 초기화
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0) # 첫 번째 Conv 레이어
        self.relu1 = nn.ReLU() # 첫 번째 ReLU 활성화 함수
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) # 첫 번째 MaxPooling 레이어
        
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0) # 두 번째 Conv 레이어
        self.relu2 = nn.ReLU() # 두 번째 ReLU 활성화 함수
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) # 두 번째 MaxPooling 레이어

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 53 * 53, 512)  # FC1, 224x224 입력 기준 feature map 크기
        self.fc2 = nn.Linear(512, num_classes) # FC2, 클래스 수

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.cnn1(x))) # Conv1 → ReLU → MaxPool1
        out = self.maxpool2(self.relu2(self.cnn2(out))) # Conv2 → ReLU → MaxPool2
        
        out = out.view(out.size(0), -1)  # Flatten 4D tensor to 2D tensor(vector) # view ( tensor의 배치 크기(size(0)), 나머지 차원은 자동 계산(-1) )
        out = F.relu(self.fc1(out))  # FC1 → ReLU
        out = self.fc2(out) # FC2 → logit 출력 
        return out  # CrossEntropyLoss 사용 → Softmax 불필요

def count_parameters(model):
    """학습 가능한 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) # requires_grad=True만 합산

def create_model(device):
    """모델 생성 후 디바이스로 이동, 파라미터 수 출력"""
    model = LeNet().to(device)
    print(f'Trainable parameters: {count_parameters(model):,}')
    return model
