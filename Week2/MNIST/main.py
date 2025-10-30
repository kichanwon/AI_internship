#!/usr/bin/env python3
"""
MNIST 손글씨 숫자 분류 - 메인 실행 스크립트
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os

# 사용자 정의 모듈 import
import config
from dataset import load_mnist_data
from dataloader import create_dataloaders, get_test_dataloader
from model import create_model
from trainer import train_model
from evaluator import (
    evaluate_model,
    plot_confusion_matrix,
    print_classification_report,
    visualize_predictions
)
def main():
    """메인 실행 함수"""
    
    # 디바이스 설정
    print(f"Using device: {config.DEVICE}")
    
    # 1. 데이터 로드
    print("\n[1단계] 데이터 로드 중...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_data(
        config.DATA_PATH,
        val_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    print(f"훈련 데이터: {len(x_train)}개")
    print(f"검증 데이터: {len(x_val)}개")
    print(f"테스트 데이터: {len(x_test)}개")
    
    # 2. 데이터로더 생성
    print("\n[2단계] 데이터로더 생성 중...")
    dataloader_dict = create_dataloaders(
        x_train, y_train, x_val, y_val,
        config.MEAN, config.STD,
        config.BATCH_SIZE
    )

    test_dataloader = get_test_dataloader(
        x_test, y_test,
        config.MEAN, config.STD,
        config.BATCH_SIZE
    )

    # 3. 모델 생성
    print("\n[3단계] 모델 생성 중...")
    model = create_model(config.DEVICE, config.NUM_CLASSES)
    
    # 4. 손실 함수 및 옵티마이저 설정
    print("\n[4단계] 학습 설정 중...")
    criterion = nn.CrossEntropyLoss().to(config.DEVICE)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM
    )
    
    train_loader = dataloader_dict["train"]
    num_steps_per_epoch = len(train_loader)
    total_steps = num_steps_per_epoch * config.NUM_EPOCHS

    print(f"손실 함수: CrossEntropyLoss")
    print(f"옵티마이저: SGD (lr={config.LEARNING_RATE}, momentum={config.MOMENTUM})")
    print(f"배치 크기: {config.BATCH_SIZE}")
    print(f"에포크 수: {config.NUM_EPOCHS}")
    print(f"스텝 수(에포크당): {num_steps_per_epoch}")
    print(f"총 학습 스텝 수: {total_steps}")

    # 5. 모델 학습
    print("\n[5단계] 모델 학습 시작...")
    trained_model, best_val_acc = train_model(
        model, dataloader_dict, criterion, optimizer,
        config.NUM_EPOCHS, config.DEVICE, config.TENSORBOARD_LOG_DIR
    )

    # 6. 최고 성능 모델 저장
    print("\n[6단계] 모델 저장 중...")
    model_path = os.path.join(config.RESULT_DIR, "best_mnist_model.pth")
    torch.save(trained_model.state_dict(), model_path)
    print(f"model_path: '{model_path}'")
    
    # 7. 테스트 데이터 평가
    print("\n[7단계] 테스트 데이터 평가 중...")
    test_acc, y_pred, y_true = evaluate_model(
        trained_model, test_dataloader, config.DEVICE
    )
        
    print(f"\n{'='*60}")
    print(f"최종 결과")
    print(f"{'='*60}")
    print(f"최고 검증 정확도: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"테스트 정확도: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'='*60}")
    
    # 8. 혼동 행렬 생성
    print("\n[8단계] 혼동 행렬 생성 중...")
    cm_path = os.path.join(config.RESULT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, config.CLASSES, cm_path)
    
    # 9. 분류 리포트 출력
    print_classification_report(y_true, y_pred, config.CLASSES)
    
    # 10. 예측 결과 시각화
    print("\n[9단계] 예측 결과 시각화 중...")
    vis_path = os.path.join(config.RESULT_DIR, "predictions_visualization.png")
    visualize_predictions(
        trained_model, test_dataloader, config.DEVICE,
        config.CLASSES, vis_path, num_images=20
    )
        
    print("=" * 60)
    print("Fin")
    print(f"'{config.RESULT_DIR}'")

if __name__ == "__main__":
    main()