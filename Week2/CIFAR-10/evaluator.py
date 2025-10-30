import os
import cv2
import math
import random
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transforms import ImageTransform

def predict_on_test_data(model, test_files, image_size, mean, std, device):
    """테스트 데이터에 대한 예측을 수행하는 함수"""
    id_list = []
    pred_list = []
    
    transform = ImageTransform(image_size, mean, std)
    
    with torch.no_grad():
        model.eval()
        for test_path in tqdm(test_files, desc="Predicting"):
            img = Image.open(test_path)
            _id = os.path.basename(test_path).split('.')[0]
            
            img = transform(img, phase='val')
            img = img.unsqueeze(0).to(device)
            
            outputs = model(img)
            preds = F.softmax(outputs, dim=1)[:, 1].tolist()
            
            id_list.append(_id)
            pred_list.append(preds[0])
    
    results = pd.DataFrame({
        'id': id_list,
        'label': pred_list
    })
    
    results.sort_values(by='id', inplace=True)
    results.reset_index(drop=True, inplace=True)
    
    return results

def create_confusion_matrix(model, test_files, image_size, mean, std, device, save_path="confusion_matrix.png"):
    """혼동행렬을 생성하고 시각화하는 함수"""
    true_labels = []
    pred_labels = []
    
    transform = ImageTransform(image_size, mean, std)  # 검증용 변환 적용
    
    with torch.no_grad():
        model.eval()
        for test_path in tqdm(test_files, desc="Creating confusion matrix"):
            img = Image.open(test_path)
            img = transform(img, phase='val') # 이미지 변환
            img = img.unsqueeze(0).to(device) # 배치 차원 추가 후 device로 이동
            
            outputs = model(img) # 모델 예측
            pred = torch.argmax(outputs, 1).item() # 예측 클래스 추출
            
            # 실제 라벨은 폴더명 기준
            true_label = 1 if "dog" in test_path else 0 # dog=1, cat=0
            
            true_labels.append(true_label)
            pred_labels.append(pred)
    
    # 혼동행렬 계산
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.show()
    
    return cm, true_labels, pred_labels

def display_predictions(test_files, predictions_df, classes, n=10, save_path="predictions_visualization.png"):
    """예측 결과를 시각화하는 함수"""
    
    # 실제 라벨별로 이미지 분리
    true_cats = [path for path in test_files if 'cat' in path]
    true_dogs = [path for path in test_files if 'dog' in path]
    
    # 각각에서 n개씩 무작위 선택
    random.seed(42)
    selected_true_cats = random.sample(true_cats, min(n, len(true_cats)))
    selected_true_dogs = random.sample(true_dogs, min(n, len(true_dogs)))
    
    # 시각화
    fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(3*n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)
    
    # 실제 고양이들의 예측 결과
    for i, filepath in enumerate(selected_true_cats):
        _id = os.path.basename(filepath).split('.')[0]
        pred_row = predictions_df[predictions_df['id'] == _id]
        
        if not pred_row.empty:
            pred_prob = pred_row['label'].iloc[0]
            pred_class = 1 if pred_prob >= 0.5 else 0
            
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 예측이 맞는지 확인
            is_correct = (pred_class == 0)  # 실제로 고양이
            color = "green" if is_correct else "red"
            
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"True: Cat\nPred: {classes[pred_class]} ({pred_prob:.3f})", 
                               color=color, fontsize=10)
            axes[0, i].axis("off")
    
    # 실제 강아지들의 예측 결과
    for i, filepath in enumerate(selected_true_dogs):
        _id = os.path.basename(filepath).split('.')[0]
        pred_row = predictions_df[predictions_df['id'] == _id]
        
        if not pred_row.empty:
            pred_prob = pred_row['label'].iloc[0]
            pred_class = 1 if pred_prob >= 0.5 else 0
            
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 예측이 맞는지 확인
            is_correct = (pred_class == 1)  # 실제로 강아지
            color = "green" if is_correct else "red"
            
            axes[1, i].imshow(img)
            axes[1, i].set_title(f"True: Dog\nPred: {classes[pred_class]} ({pred_prob:.3f})", 
                               color=color, fontsize=10)
            axes[1, i].axis("off")
    
    plt.suptitle("Test Dataset Predictions (Green=Correct, Red=Wrong)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"실제 고양이 이미지: {len(selected_true_cats)}개")
    print(f"실제 강아지 이미지: {len(selected_true_dogs)}개")
    print(f"결과 이미지가 '{save_path}'에 저장되었습니다.")