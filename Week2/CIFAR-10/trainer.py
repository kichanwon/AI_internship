import time
import torch
import config
from tqdm import tqdm

def train_model(model, dataloader_dict, criterion, optimizer, num_epochs, device, accum_steps=config.ACCUMULATION_STEPS):
    """
    모델 학습 함수

    Args:
        model : 학습할 nn.Module 모델
        dataloader_dict : {'train': train_loader, 'val': val_loader}
        criterion : 손실 함수
        optimizer : 옵티마이저
        num_epochs : 학습 epoch 수
        device : 'cuda' 또는 'cpu'
        accum_steps : gradient accumulation step 수
    """
    
    since = time.time() # 학습 시작 시간 기록
    best_acc = 0.0 # 최고 정확도 초기화
    best_model_wts = None # 최고 성능 모델 가중치 초기화
    
    scaler = torch.amp.GradScaler(device=config.DEVICE.type)  # Mixed Precision용 스케일러

    gpu_usage = []  # GPU 메모리 사용량 기록 리스트 (MB)

    for epoch in range(num_epochs): 
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)
        
        for phase in ['train', 'val']: # 학습/검증 단계
            if phase == 'train':
                model.train() # 학습 모드
            else:
                model.eval() # 평가 모드
            
            epoch_loss = 0.0 # epoch 손실 초기화
            epoch_corrects = 0 # epoch 맞춘 샘플 수 초기화

            # 학습 단계에서만 gradient 초기화
            if phase == 'train':
                optimizer.zero_grad()

            for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader_dict[phase], desc=f'{phase} phase')):
                inputs = inputs.to(device) # 입력을 device로 이동
                labels = labels.to(device) # 라벨을 device로 이동
                
                with torch.set_grad_enabled(phase == 'train'):  # 학습 단계에만 gradient 계산
                    with torch.amp.autocast(device_type=config.DEVICE.type):  # Mixed Precision 적용
                        outputs = model(inputs)  # 모델 예측
                        _, preds = torch.max(outputs, 1) # 예측 클래스 추출
                        loss = criterion(outputs, labels) / accum_steps  # 손실 계산(Accumulation 반영)
                    
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        
                        if (batch_idx + 1) % accum_steps == 0:
                            scaler.step(optimizer) # epoch / step
                            scaler.update()
                            optimizer.zero_grad()

                # GPU 메모리 사용량 기록 (MB)          
                if device.type == 'cuda':
                    gpu_usage.append(torch.cuda.memory_allocated(device)/1024**2)

                epoch_loss += loss.item() * inputs.size(0) * accum_steps # 원래 loss로 복원 # 배치 손실 누적
                epoch_corrects += torch.sum(preds == labels.data) # 맞춘 샘플 수 누적
            
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset) # 평균 손실 계산
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset) # 정확도 계산
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc: # 최고 성능 갱신
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()
        
        print()
    
    # GPU 사용 통계 계산
    if gpu_usage:
        print(f'GPU Memory Usage (MB) - Max: {max(gpu_usage):.1f}, Min: {min(gpu_usage):.1f}, Avg: {sum(gpu_usage)/len(gpu_usage):.1f}')
    
    time_elapsed = time.time() - since # 학습 시간 계산
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # 최고 성능 모델 가중치 로드
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    return model, best_acc