import torch
import os

# 디바이스 설정
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # CUDA 사용 가능 시 첫 번째 GPU, 아니면 CPU 선택

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 파일 기준 디렉토리
RESULT_DIR = os.path.join(BASE_DIR, "result") # 결과 저장 디렉토리
os.makedirs(RESULT_DIR, exist_ok=True) # 디렉토리 없으면 생성

# 데이터 경로
CAT_DIRECTORY = "/home/user3/data/CatAndDog/cat" 
DOG_DIRECTORY = "/home/user3/data/CatAndDog/dog"

# 이미지 전처리 설정
IMAGE_SIZE = 224 # 이미지 크기
MEAN = (0.485, 0.456, 0.406) # 이미지 정규화용 평균값
STD = (0.229, 0.224, 0.225) # 이미지 정규화용 표준편차값

# =========================================
# 학습 설정
# =========================================
"""
NUM_EPOCHS 3 / LR = 0.001 기준

ACCUMULATION_STEPS = 8 
    BATCH_SIZE  Max     Min     Avg     ACC     train complete in
    64          581.2   405.1   560.6   0.5860  2m 37s
    128         618.0   441.9   593.2   0.5990  2m 28s
    256         692.0   516.4   659.3   0.5520  2m 34s
    512         837.6   561.9   784.8   0.5090  2m 52s


ACCUMULATION_STEPS = 1
    BATCH_SIZE  Max     Min     Avg     ACC     train complete in
    128         441.9   372.5   440.2   0.6340  3m 2s
    128         441.9   372.5   440.2   0.6310  2m 56s
    512         661.9   386.3   639.9   0.5960  2m 52s  
    

    BATCH_SIZE  Max     Min     Avg     ACC     train complete in
ACCUMULATION_STEPS = 1 / LR = 0.001
    128         441.9   372.5   440.2   0.6310  2m 56s
ACCUMULATION_STEPS = 4 / LR = 0.001*4
    128         618.0   441.9   579.3   0.5780  2m 45s


- 정확도를 올리려면 gpu를 활용하지 못해 속도 저하 
- gpu 활용을 위해 batch를 누적하면 정확도가 저하
학습률·정규화·스케줄링 조정(예: cosine LR, warmup, weight decay) 등을 병행하면 큰 batch에서도 정확도 하락을 완화 가능
"""

# 배치 크기
BATCH_SIZE = 64

# Gradient Accumulation 단계 수
ACCUMULATION_STEPS = 1
"""
Gradient Accumulation(경사 누적) 기능:
- 작은 배치 여러 개의 gradient를 누적해 두었다가 일정 step마다 가중치 업데이트
- 메모리가 부족해 큰 배치를 직접 GPU에 올릴 수 없을 때 사용

예시)
BATCH_SIZE = 128
ACCUMULATION_STEPS = 8

→ 실제 GPU에 올려 처리하는 배치 크기 = 128
→ 8번 gradient 누적 후 optimizer.step() 실행
→ 최종적으로 Effective Batch Size = 128 × 8 = 1024

장점:
- GPU 메모리 부족 문제 해결
- 큰 배치 효과를 흉내낼 수 있어 gradient 안정화에 도움

단점:
- 업데이트 주기가 늦어져 학습 속도가 다소 느려질 수 있음
- 너무 큰 Effective Batch는 일반화 성능을 떨어뜨릴 수 있음
"""


NUM_EPOCHS = 3
    # 데이터 반복 학습 횟수 / 너무 많으면 과적합, 너무 적으면 미학습
# 학습률
LEARNING_RATE = 0.001*4
    # 모델 가중치 업데이트 스텝 크기 / 너무 크면 발산, 너무 작으면 느림
# 옵티마이저 모멘텀 (SGD)
MOMENTUM = 0.9
    # 관성 개념 도입 / 너무 크면 발산, 너무 작으면 느림 / 과거 gradient 반영 → 학습 안정화

# 데이터 분할 비율
TEST_SIZE = 0.1 # 테스트 데이터 비율
VAL_SIZE = 0.2 # 검증 데이터 비율
RANDOM_STATE = 42 # 랜덤 시드

# 클래스 라벨
CLASSES = {0: 'cat', 1: 'dog'} # 0: 고양이, 1: 강아지