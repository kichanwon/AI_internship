import torch
import os

# 디바이스 설정
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # CUDA 사용 가능 시 첫 번째 GPU, 아니면 CPU 선택

# 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 파일 기준 디렉토리
RESULT_DIR = os.path.join(BASE_DIR, "result") # 결과 저장 디렉토리
os.makedirs(RESULT_DIR, exist_ok=True) # 디렉토리 없으면 생성

# TensorBoard 설정
TENSORBOARD_LOG_DIR = os.path.join(os.path.dirname(BASE_DIR), "logs")  # TensorBoard 로그 폴더
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)  # 로그 폴더 생성

# WandB 설정
WANDB_ENABLED = True  # WandB 사용 여부
# WANDB_ENTITY = "your-team-name"  # WandB 팀 이름 (개인 계정이면 None)
WANDB_PROJECT = "ResNet_implement"  # 프로젝트 이름

# 데이터 경로
DATA_PATH = '/home/user3/data/cifar-python/cifar-10-batches-py'

if os.path.exists(DATA_PATH):
    print(f"데이터 파일 발견: {DATA_PATH}")
else:
    raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")

# 이미지 설정
IMAGE_SIZE = 32  # CIFAR-10: 32x32
MEAN = (0.4914, 0.4822, 0.4465)  # CIFAR-10 RGB 평균
STD = (0.2470, 0.2435, 0.2616)   # CIFAR-10 RGB 표준편차

# Adam 옵티마이저 설정
ADAM_BETAS = (0.9, 0.999)  # beta1, beta2
ADAM_WEIGHT_DECAY = 0.0  # L2 정규화

# =========================================
# 학습 설정
# =========================================

# 활성화 함수 설정
ACTIVATION_FUNCTION = 'tanh'  # 'relu', 'tanh'

# 학습 하이퍼파라미터
BATCH_SIZE = 512  # 한 번에 처리할 데이터 개수
NUM_EPOCHS = 100   # 전체 데이터를 몇 번 반복 학습할지
LEARNING_RATE = 0.005  # 학습률 (가중치 업데이트 크기)
# MOMENTUM = 0.9   # SGD 옵티마이저의 모멘텀

# 데이터 분할 비율
# TEST_SIZE = 0.1 # 테스트 데이터 비율
VAL_SIZE = 0.2 # 검증 데이터 비율
RANDOM_STATE = 42 # 랜덤 시드

# 클래스 라벨
NUM_CLASSES = 10
CLASSES = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
    5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}


# =========================================
# tensorboard --logdir=Week2/logs --port=6006