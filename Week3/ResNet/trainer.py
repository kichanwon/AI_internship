import time
import torch
import config
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import wandb

def train_model(model, dataloader_dict, criterion, optimizer, num_epochs, device, log_dir):
    """
    모델 학습 함수
        Args:
            model: 학습할 모델
            dataloader_dict: {'train': train_loader, 'val': val_loader}
            criterion: 손실 함수 (예: CrossEntropyLoss)
            optimizer: 옵티마이저 (예: SGD, Adam)
            num_epochs: 학습 반복 횟수
            device: 'cuda' 또는 'cpu'
        Returns:
            model: 학습된 모델
            best_acc: 최고 검증 정확도
    """
    
    # TensorBoard Writer 생성
    # 실행 시간을 폴더명에 포함하여 구분
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    optimizer_name = optimizer.__class__.__name__
    model_name = model.__class__.__name__
    experiment_name = (
        f"cifar-10/"
        f"{model_name}/"
        f"activation-{config.ACTIVATION_FUNCTION}/"
        f"optimizer-{optimizer_name}/"
        f"lr{config.LEARNING_RATE}_bs{config.BATCH_SIZE}_ep{config.NUM_EPOCHS}/"
        f"{timestamp}/"
    )
    writer = SummaryWriter(os.path.join(log_dir, experiment_name))
    print(f"log saved: {writer.log_dir}")
    print(f"tensorboard --logdir={log_dir}")

    # WandB 초기화
    if config.WANDB_ENABLED:
            wandb_run = wandb.init(
                entity=config.WANDB_ENTITY if hasattr(config, 'WANDB_ENTITY') else None,
                project=config.WANDB_PROJECT if hasattr(config, 'WANDB_PROJECT') else "cifar10-experiments",
                name=f"{model_name}_{config.ACTIVATION_FUNCTION}_{timestamp}",
                config={
                    "learning_rate": config.LEARNING_RATE,
                    "architecture": model_name,
                    "activation": config.ACTIVATION_FUNCTION,
                    "dataset": "CIFAR-10",
                    "epochs": num_epochs,
                    "batch_size": config.BATCH_SIZE,
                    "optimizer": optimizer_name,
                    "adam_betas": config.ADAM_BETAS,
                    "weight_decay": config.ADAM_WEIGHT_DECAY,
                },
                tags=[model_name, config.ACTIVATION_FUNCTION, optimizer_name]
            )
            # 모델 구조를 WandB에 기록
            wandb.watch(model, criterion, log="all", log_freq=100)
            print(f"WandB run initialized: {wandb_run.name}")


    since = time.time() # 학습 시작 시간 기록
    best_acc = 0.0 # 최고 정확도 초기화
    best_model_wts = None # 최고 성능 모델 가중치 초기화
    
    # 학습 기록용 리스트
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    global_step = 0

    scaler = torch.amp.GradScaler(device=config.DEVICE.type)  # Mixed Precision용 스케일러

    gpu_usage = []  # GPU 메모리 사용량 기록 리스트 (MB)

    for epoch in range(num_epochs): 
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 30)
        
        for phase in ['train', 'val']: # 학습/검증 단계
            if phase == 'train':
                model.train() # 학습 모드
            else:
                model.eval() # 평가 모드
            
            running_loss = 0.0  # 누적 손실
            running_corrects = 0  # 누적 정답 개수

            for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader_dict[phase], desc=f'{phase} phase')):
                # 데이터를 device로 이동
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 학습 단계에서만 gradient 초기화
                if phase == 'train':
                    optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):  # 학습 단계에만 gradient 계산
                    with torch.amp.autocast(device_type=config.DEVICE.type):  # Mixed Precision 적용
                        outputs = model(inputs)  # 모델 예측
                        _, preds = torch.max(outputs, 1) # 예측 클래스 추출
                        loss = criterion(outputs, labels) # 손실 계산
                    
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer) # epoch / step
                        scaler.update()
                        writer.add_scalar('Batch/Train_Loss', loss.item(), global_step)

                        # WandB에 배치별 손실 기록
                        if config.WANDB_ENABLED:
                            wandb.log({"batch_train_loss": loss.item()}, step=global_step)

                        global_step += 1

                # GPU 메모리 사용량 기록          
                if device.type == 'cuda':
                    gpu_usage.append(torch.cuda.memory_allocated(device)/1024**2)

                running_loss += loss.item() * inputs.size(0) # 배치 손실 누적
                running_corrects += torch.sum(preds == labels.data) # 맞춘 샘플 수 누적
            
            running_loss = running_loss / len(dataloader_dict[phase].dataset) # 평균 손실 계산
            running_acc = running_corrects.double() / len(dataloader_dict[phase].dataset) # 정확도 계산
            print(f'{phase} Loss: {running_loss:.4f} Acc: {running_acc:.4f}')

            # TensorBoard에 에포크별 기록
            if phase == 'train':
                train_loss_list.append(running_loss)
                train_acc_list.append(running_acc.item())
                # 스칼라 기록
                writer.add_scalar('Epoch/Train_Loss', running_loss, epoch)
                writer.add_scalar('Epoch/Train_Accuracy', running_acc, epoch)
                # 학습률 기록
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch)
                # WandB에 기록
                if config.WANDB_ENABLED:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": running_loss,
                        "train_accuracy": running_acc.item(),
                        "learning_rate": current_lr,
                    })
            else:
                # 검증 손실/정확도 기록
                val_loss_list.append(running_loss)
                val_acc_list.append(running_acc.item())
                # 스칼라 기록
                writer.add_scalar('Epoch/Val_Loss', running_loss, epoch)
                writer.add_scalar('Epoch/Val_Accuracy', running_acc, epoch)
                # WandB에 기록
                if config.WANDB_ENABLED:
                    wandb.log({
                        "epoch": epoch + 1,
                        "val_loss": running_loss,
                        "val_accuracy": running_acc.item(),
                    })

            if phase == 'val' and running_acc > best_acc: # 최고 성능 갱신
                best_acc = running_acc
                best_model_wts = model.state_dict().copy()
                
        # 에포크마다 Train vs Val 비교 그래프
        writer.add_scalars('Comparison/Loss', {
            'Train': train_loss_list[-1],
            'Val': val_loss_list[-1]
        }, epoch)
        
        writer.add_scalars('Comparison/Accuracy', {
            'Train': train_acc_list[-1],
            'Val': val_acc_list[-1]
        }, epoch)

    # 학습 완료 정보 출력
    time_elapsed = time.time() - since
    print('\n' + '=' * 40)
    print(f'time cost: {time_elapsed//60:.0f}분 {time_elapsed%60:.0f}초')
    print(f'best acc: {best_acc:.4f}')
    print('=' * 40)

    # WandB에 최종 결과 기록
    if config.WANDB_ENABLED:
        wandb.log({
            "best_val_accuracy": best_acc.item(),
            "final_train_loss": train_loss_list[-1],
            "final_val_loss": val_loss_list[-1],
            "training_time_minutes": time_elapsed / 60,
        })

    # 최종 하이퍼파라미터 기록
    writer.add_hparams(
        {
            'lr': optimizer.param_groups[0]['lr'],
            'batch_size': dataloader_dict['train'].batch_size,
            'epochs': num_epochs,
        },
        {
            'hparam/best_accuracy': best_acc.item(),
            'hparam/final_train_loss': train_loss_list[-1],
            'hparam/final_val_loss': val_loss_list[-1],
        }
    )

    # 모델 그래프 기록 (첫 번째 배치로)
    sample_input, _ = next(iter(dataloader_dict['train']))
    writer.add_graph(model, sample_input.to(device))
    
    # Writer 종료
    writer.close()
    print(f"\nTensorBoard log saved: {writer.log_dir}")


    # GPU 사용 통계 계산 및 WandB 기록
    if gpu_usage:
        gpu_max = max(gpu_usage)
        gpu_min = min(gpu_usage)
        gpu_avg = sum(gpu_usage)/len(gpu_usage)
        print(f'GPU Memory Usage (MB) - Max: {gpu_max:.1f}, Min: {gpu_min:.1f}, Avg: {gpu_avg:.1f}')
        
        if config.WANDB_ENABLED:
            wandb.log({
                "gpu_memory_max_mb": gpu_max,
                "gpu_memory_min_mb": gpu_min,
                "gpu_memory_avg_mb": gpu_avg,
            })

    time_elapsed = time.time() - since # 학습 시간 계산
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # 최고 성능 모델 가중치 로드
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # WandB run 종료
    if config.WANDB_ENABLED:
        wandb.finish()
        print("WandB run finished")

    return model, best_acc