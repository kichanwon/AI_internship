import torch
import os

# 디바이스 설정
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # CUDA 사용 가능 시 첫 번째 GPU, 아니면 CPU 선택
print(f"Using device: {DEVICE}")

# 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 파일 기준 디렉토리
RESULT_DIR = os.path.join(BASE_DIR, "result") # 결과 저장 디렉토리
os.makedirs(RESULT_DIR, exist_ok=True) # 디렉토리 없으면 생성
TENSORBOARD_LOG_DIR = os.path.join(os.path.dirname(BASE_DIR), "logs")  # TensorBoard 로그 폴더
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)  # 로그 폴더 생성


# 데이터 경로
# DATA_PATH = '../data/mnist.npz' # 실행하는 터미널 기준임!!
DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'mnist.npz')

if os.path.exists(DATA_PATH):
    print(f"데이터 파일 발견: {DATA_PATH}")
else:
    raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")

# 이미지 설정
IMAGE_SIZE = 28  # MNIST 이미지 크기 (28x28)
MEAN = (0.1307,)  # MNIST 데이터셋의 평균값 (흑백 이미지이므로 1채널)
STD = (0.3081,)   # MNIST 데이터셋의 표준편차

# Adam 옵티마이저 설정
ADAM_BETAS = (0.9, 0.999)  # beta1, beta2
ADAM_WEIGHT_DECAY = 0.0  # L2 정규화

# =========================================
# 학습 설정
# =========================================

# 학습 하이퍼파라미터
BATCH_SIZE = 64  # 한 번에 처리할 데이터 개수
NUM_EPOCHS = 5   # 전체 데이터를 몇 번 반복 학습할지
LEARNING_RATE = 0.001  # 학습률 (가중치 업데이트 크기)
# MOMENTUM = 0.9   # SGD 옵티마이저의 모멘텀

# 데이터 분할 비율
# TEST_SIZE = 0.1 # 테스트 데이터 비율
VAL_SIZE = 0.2 # 검증 데이터 비율
RANDOM_STATE = 42 # 랜덤 시드

# 클래스 라벨
NUM_CLASSES = 10  # MNIST는 0~9까지 10개 숫자
CLASSES = {i: str(i) for i in range(10)}  # {0: '0', 1: '1', ..., 9: '9'}



# =========================================
# tensorboard --logdir=Week2/MNIST/logs --port=6006