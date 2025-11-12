import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tqdm import tqdm

def evaluate_model(model, dataloader, device):
    """
    모델 평가 함수
    
    Args:
        model: 평가할 모델
        dataloader: 테스트 데이터로더
        device: 'cuda' 또는 'cpu'
    
    Returns:
        accuracy: 정확도
        all_preds: 모든 예측값
        all_labels: 모든 실제 레이블
    """
    model.eval()  # 평가 모드로 전환
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # 그래디언트 계산 비활성화 (메모리 절약)
        for inputs, labels in tqdm(dataloader, desc="평가 중"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 예측
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # 결과 저장
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 정확도 계산
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = np.mean(all_preds == all_labels)
    
    return accuracy, all_preds, all_labels