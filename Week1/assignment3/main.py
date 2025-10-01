#!/usr/bin/env python3
"""
Dog vs Cat Classification - Main Training Script
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os

# 사용자 정의 모듈 import
import config
from dataset import prepare_data
from dataloader import create_dataloaders
from model import create_model
from trainer import train_model
from evaluator import predict_on_test_data, create_confusion_matrix, display_predictions

def main():
    """메인 실행 함수"""
    
    # 디바이스 설정
    print(f"Using device: {config.DEVICE}")
    
    # 1. 데이터 준비
    print("\n=== 데이터 준비 중... ===")
    train_files, val_files, test_files = prepare_data(
        config.CAT_DIRECTORY, 
        config.DOG_DIRECTORY,
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    print(f"훈련 데이터: {len(train_files)}개")
    print(f"검증 데이터: {len(val_files)}개")
    print(f"테스트 데이터: {len(test_files)}개")
    
    # 2. 데이터로더 생성
    print("\n=== 데이터로더 생성 중... ===")
    dataloader_dict = create_dataloaders(
        train_files, val_files,
        config.IMAGE_SIZE, config.MEAN, config.STD,
        config.BATCH_SIZE
    )
    
    # 3. 모델 생성
    print("\n=== 모델 생성 중... ===")
    model = create_model(config.DEVICE)
    
    # 4. 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss().to(config.DEVICE)
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        momentum=config.MOMENTUM
    )
    
    # 5. 모델 학습
    print("\n=== 모델 학습 시작... ===")
    trained_model, best_acc = train_model(
        model, dataloader_dict, criterion, optimizer, 
        config.NUM_EPOCHS, config.DEVICE
    )

    # 5-1. 최고 성능 모델 저장
    model_path = os.path.join(config.RESULT_DIR, "best_model.pth")
    torch.save(trained_model.state_dict(), model_path)
    print(f"최고 성능 모델이 '{model_path}'에 저장되었습니다.")
    
    # 6. 테스트 데이터 예측
    print("\n=== 테스트 데이터 예측 중... ===")
    predictions = predict_on_test_data(
        trained_model, test_files,
        config.IMAGE_SIZE, config.MEAN, config.STD,
        config.DEVICE
    )
    
    # 예측 결과 저장
    predictions.to_csv(os.path.join(config.RESULT_DIR, 'LeNet_predictions.csv'), index=False)
    print("예측 결과가 'LeNet_predictions.csv'에 저장되었습니다.")
    
    # 7. 혼동행렬 생성
    print("\n=== 혼동행렬 생성 중... ===")
    cm, true_labels, pred_labels = create_confusion_matrix(
        trained_model, test_files,
        config.IMAGE_SIZE, config.MEAN, config.STD,
        config.DEVICE,
        save_path=os.path.join(config.RESULT_DIR, "confusion_matrix.png")
    )
    
    # 8. 예측 결과 시각화
    print("\n=== 예측 결과 시각화 중... ===")
    display_predictions(
        test_files, predictions, config.CLASSES,
        n=10, save_path=os.path.join(config.RESULT_DIR, "predictions_by_true_labels.png")
    )
    
    # 9. 최종 결과 출력
    print(f"\n=== 최종 결과 ===")
    print(f"최고 검증 정확도: {best_acc:.4f}")
    
    # 테스트 정확도 계산
    correct_predictions = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred)
    test_accuracy = correct_predictions / len(true_labels)
    print(f"테스트 정확도: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()