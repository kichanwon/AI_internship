import config
import os
import torch

from dataset import prepare_data
from dataloader import get_test_dataloader
from transforms import ImageTransform
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm

# 모델 클래스 불러오기
from model import LeNet
from evaluator import predict_on_test_data

# 테스트 파일 준비
train_files, val_files, test_files = prepare_data(
    config.CAT_DIRECTORY,
    config.DOG_DIRECTORY,
    test_size=config.TEST_SIZE,
    val_size=config.VAL_SIZE,
    random_state=config.RANDOM_STATE
)

# 모델 생성
model = LeNet().to(config.DEVICE)

# 저장된 최고 성능 모델 불러오기
model_path = os.path.join(config.RESULT_DIR, "best_model.pth")
model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
model.eval()

# 테스트 데이터 예측
predictions = predict_on_test_data(
    model, test_files,
    config.IMAGE_SIZE, config.MEAN, config.STD,
    config.DEVICE
)

# 결과 CSV 저장
predictions.to_csv(os.path.join(config.RESULT_DIR, 'best_model_predictions.csv'), index=False)
print("테스트 데이터 예측 결과가 'best_model_predictions.csv'에 저장되었습니다.")
print(predictions)