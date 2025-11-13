import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tqdm import tqdm
import wandb

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
    running_loss = 0.0
 
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

def create_confusion_matrix(model, dataloader, device, class_names, save_path="confusion_matrix.png"):
    """
    혼동행렬을 생성하고 시각화하는 함수
    
    Args:
        model: 평가할 모델
        dataloader: 데이터로더
        device: 'cuda' 또는 'cpu'
        class_names: 클래스 이름 리스트 (예: ['airplane', 'automobile', ...])
        save_path: 저장할 경로
    
    Returns:
        cm: 혼동행렬
        true_labels: 실제 레이블 리스트
        pred_labels: 예측 레이블 리스트
    """
    true_labels = []
    pred_labels = []
    
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Creating confusion matrix"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 모델 예측
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            
            # 결과 저장
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
    
    # 혼동행렬 계산
    cm = confusion_matrix(true_labels, pred_labels)
    wandb.log({"confusion_matrix": wandb.Image(save_path)})

    # 시각화
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    plt.title("Confusion Matrix", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"혼동행렬이 '{save_path}'에 저장되었습니다.")
    
    return cm, true_labels, pred_labels


def print_classification_report(true_labels, pred_labels, class_names):
    """
    분류 리포트 출력
    
    Args:
        true_labels: 실제 레이블
        pred_labels: 예측 레이블
        class_names: 클래스 이름 리스트
    """
    print("\n" + "="*60)
    print("분류 리포트 (Classification Report)")
    print("="*60)
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=class_names,
        digits=4
    )
    print(report)

    # WandB에 분류 리포트 로그
    if wandb.run is not None:
        wandb.sklearn.classification_report(
            y_true=true_labels,
            y_pred=pred_labels,
            labels=range(len(class_names)),
            target_names=class_names
        )
