from torch.utils.data import DataLoader
from dataset import DogvsCatDataset
from transforms import ImageTransform

def create_dataloaders(train_files, val_files, image_size, mean, std, batch_size):
    """학습/검증용 데이터셋 → 데이터로더 생성"""
    
    # 이미지 변환기 생성
    transform = ImageTransform(image_size, mean, std)
    
    # 데이터셋 생성
    train_dataset = DogvsCatDataset(train_files, transform=transform, phase='train')
    val_dataset = DogvsCatDataset(val_files, transform=transform, phase='val')
    
    # 데이터로더 생성
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}
    
    return dataloader_dict

def get_test_dataloader(test_files, image_size, mean, std, batch_size=1):
    """테스트용 데이터셋 → 데이터로더 생성"""
    
    transform = ImageTransform(image_size, mean, std) # 테스트 시에도 검증용 변환 적용
    test_dataset = DogvsCatDataset(test_files, transform=transform, phase='val') # phase='val'로 고정
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_dataloader