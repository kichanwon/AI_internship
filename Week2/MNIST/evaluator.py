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


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """
    혼동 행렬(Confusion Matrix) 시각화
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        classes: 클래스 이름 딕셔너리
        save_path: 저장 경로
    """
    # 혼동 행렬 계산
    cm = confusion_matrix(y_true, y_pred)
    
    # 시각화
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[classes[i] for i in range(len(classes))]
    )
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    
    plt.title('Confusion Matrix - MNIST Classification', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Confusion Matrix: '{save_path}'")


def print_classification_report(y_true, y_pred, classes):
    """
    분류 성능 리포트 출력
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        classes: 클래스 이름 딕셔너리
    """
    target_names = [classes[i] for i in range(len(classes))]
    report = classification_report(y_true, y_pred, target_names=target_names)
    
    print("\n" + "=" * 60)
    print("classification_report")
    print("=" * 60)
    print(report)


def visualize_predictions(model, dataloader, device, classes, save_path, num_images=20):
    """
    예측 결과 시각화
    
    Args:
        model: 모델
        dataloader: 데이터로더
        device: 디바이스
        classes: 클래스 이름
        save_path: 저장 경로
        num_images: 표시할 이미지 개수
    """
    model.eval()
    
    images_shown = 0
    fig = plt.figure(figsize=(15, 8))
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # 배치의 이미지들을 표시
            for i in range(inputs.size(0)):
                if images_shown >= num_images:
                    break
                
                ax = plt.subplot(4, 5, images_shown + 1)
                
                # 이미지 표시 (정규화 해제)
                img = inputs[i].cpu().numpy()[0]
                plt.imshow(img, cmap='gray')
                
                # 제목 설정 (정답 여부에 따라 색상 변경)
                true_label = classes[labels[i].item()]
                pred_label = classes[preds[i].item()]
                
                if labels[i] == preds[i]:
                    color = 'green'
                    title = f'✓ True: {true_label}\nPred: {pred_label}'
                else:
                    color = 'red'
                    title = f'✗ True: {true_label}\nPred: {pred_label}'
                
                plt.title(title, color=color, fontsize=9)
                plt.axis('off')
                
                images_shown += 1
            
            if images_shown >= num_images:
                break
    
    plt.suptitle('MNIST Predictions (Green=Correct, Red=Wrong)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"visualize_predictions '{save_path}'")