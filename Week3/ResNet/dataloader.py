from torch.utils.data import DataLoader
from dataset import CIFAR10Dataset
from transforms import CIFARTransform

def create_dataloaders(x_train, y_train, x_val, y_val, mean, std, batch_size):
    """
    훈련/검증 데이터로더 생성
    
    Args:
        x_train, y_train: 훈련 데이터
        x_val, y_val: 검증 데이터
        mean, std: 정규화 파라미터
        batch_size: 배치 크기
    
    Returns:
        dataloader_dict: {'train': train_loader, 'val': val_loader}
    """

    # 이미지 변환기 생성
    transform = CIFARTransform(mean, std)
    
    # 데이터셋 생성
    train_dataset = CIFAR10Dataset(x_train, y_train, transform=transform, phase='train')
    val_dataset = CIFAR10Dataset(x_val, y_val, transform=transform, phase='val')
    
    # 데이터로더 생성
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
    
    dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}
    
    return dataloader_dict

def get_test_dataloader(x_test, y_test, mean, std, batch_size=64):
    """
    테스트 데이터로더 생성
    
    Args:
        x_test, y_test: 테스트 데이터
        mean, std: 정규화 파라미터
        batch_size: 배치 크기
    
    Returns:
        test_dataloader: 테스트 데이터로더
    """

    transform = CIFARTransform(mean, std)
    test_dataset = CIFAR10Dataset(x_test, y_test, transform=transform, phase='val') # phase='val'로 고정

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
    
    return test_dataloader