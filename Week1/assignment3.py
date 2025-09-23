import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from sklearn.model_selection import train_test_split

import os
import cv2
from PIL import Image
from tqdm import tqdm
import math
import time
import random
from matplotlib import pyplot as plt

import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

# 이미지 데이터셋 전처리
class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)



# 이미지 데이터셋 불러오기
cat_directory = "/home/user3/data/CatAndDog/cats"
dog_directory = "/home/user3/data/CatAndDog/dogs"

cat_images = sorted([os.path.join(cat_directory, f) for f in os.listdir(cat_directory)])
dog_images = sorted([os.path.join(dog_directory, f) for f in os.listdir(dog_directory)])

all_images = cat_images + dog_images
labels = [0]*len(cat_images) + [1]*len(dog_images)

# 훈련/검증/평가 데이터 분할
    # 훈련+검증 / 평가 (10%)
trainval_files, test_files, trainval_labels, test_labels = train_test_split(
    all_images, labels, test_size=0.1, stratify=labels, random_state=42
)

    # 훈련 / 검증 (20%)
train_files, val_files, y_train, y_val = train_test_split(
    trainval_files, trainval_labels, test_size=0.2, stratify=trainval_labels, random_state=42
)
# print('\n===============================================================================')
# print(f"Train: {len(train_files)} / Val: {len(val_files)} / Test: {len(test_files)}")
# print(f"Train cats: {y_train.count(0)}, Train dogs: {y_train.count(1)}")
# print(f"Val cats:   {y_val.count(0)}, Val dogs:   {y_val.count(1)}")
# print(f"Test cats:  {test_labels.count(0)}, Test dogs:  {test_labels.count(1)}")


# 테스트 데이터셋 이미지 확인 함수
def test_image_grid(images_filepaths,predicted_labels=(), cols=5, save_path="test_samples.png"):
    rows = math.ceil(len(images_filepaths) / cols)
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 3*rows))
    # ax = ax.ravel()

    for i, images_filepath in enumerate(images_filepaths):
        image = cv2.imread(images_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        true_label = os.path.normpath(images_filepath).split(os.sep)[-2]
        predicted_label= predicted_labels[i] if predicted_labels else true_label

        color = "green" if true_label == predicted_label else "red"
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(predicted_label, color=color)
        ax.ravel()[i].set_axis_off()

    for j in range(i+1, len(ax.ravel())):
        ax.ravel()[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)

# 테스트 데이터셋 이미지 저장
# test_image_grid(test_files[:20])

# 이미지 데이터셋 클래스 정의
class DogvsCatDataset(Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list        # 데이터 경로 목록
        self.transform = transform        # 변환기(augmentation 포함)
        self.phase = phase                # train / val 구분

        self.labels = []
        for path in self.file_list:
            folder = os.path.basename(os.path.dirname(path)) # 폴더명 기준 추출
            label = 1 if folder == "dogs" else 0
            self.labels.append(label)


    def __len__(self):
        return len(self.file_list)        # 전체 데이터 크기 반환
    
    def __getitem__(self, index):
        img_path = self.file_list[index]  # index에 해당하는 이미지 경로
        img = Image.open(img_path)        # 이미지 로드 (PIL 객체)
        img_transformed = self.transform(img, self.phase) # 변환 적용

        label = torch.tensor(self.labels[index], dtype=torch.long)

        return img_transformed, label


# 변수 값 정의
size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch_size = 32

# 이미지 데이터셋 정의
train_dataset = DogvsCatDataset(trainval_files, transform=ImageTransform(size,mean,std), phase='train')
val_dataset = DogvsCatDataset(val_files, transform=ImageTransform(size,mean,std), phase='val')

index = 0
# print('\n===============================================================================')
# print(train_dataset.__getitem__(index)[0].size())
# print(train_dataset.__getitem__(index)[1])

# 데이터로더 정의
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}

batch_iterator = iter(train_dataloader)
inputs, label = next(batch_iterator)

# print('\n===============================================================================')
# print(inputs.size())
# print(label)

# 모델의 네트워크 클래스
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = None
        self.fc2 = None
        # self.fc1 = nn.Linear(32*53*53, 512)
        # self.relu5 = nn.ReLU()
        # self.fc2 = nn.Linear(512,2)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        # print("After Conv1 + Pool1:", out.shape)

        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        # print("After Conv2 + Pool2:", out.shape)

        out = out.view(out.size(0), -1)
        # print("After Flatten:", out.shape)

        if self.fc1 is None:
            in_features = out.size(1)
            self.fc1 = nn.Linear(in_features, 512).to(out.device)
            self.fc2 = nn.Linear(512, 2).to(out.device)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.output(out)
        return out

# 모델 객체 생성
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)
# summary(model, input_size=(3,224,224))


# 학습 가능한 파라미터 확인
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print('trainable parameters ',count_parameters(model))


# 옵티마이저와 손실 함수 정의
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()


# 모델의 파라미터와 손실 함수를 할당
model = model.to(device)
criterion = criterion.to(device)

# 모델 학습 함수 정의
def train_model(model,dataloader_dict, criterion, optimizer, numepoch):
    since = time.time()
    best_acc =0.0

    for epoch in range(numepoch):
        print('Epoch {}/{}'.format(epoch+1, num_epoch))
        print('-'*20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, labels in tqdm(dataloader_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Traing complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    return model


# 모델 학습
num_epoch = 10
model = train_model(model, dataloader_dict, criterion, optimizer, num_epoch)

# 모델을 이용한 예측
id_list = []
pred_list = []
_id = 0

with torch.no_grad():
    for test_path in tqdm(test_files):
        img = Image.open(test_path)
        _id = os.path.basename(test_path).split('.')[0]
        transform = ImageTransform(size, mean, std)
        img = transform(img, phase='val')
        img = img.unsqueeze(0)
        img = img.to(device)

        model.eval()
        outputs = model(img)
        preds = F.softmax(outputs, dim=1)[:, 1].tolist()

        id_list.append(_id)
        pred_list.append(preds[0])

res = pd.DataFrame({
    'id':id_list,
    'label':pred_list
})

res.sort_values(by='id', inplace=True)
res.reset_index(drop=True, inplace=True)

res.to_csv('LeNet', index=False)


# 테스트 데이터셋 이미지를 출력하기위한 함수 정의
class_ = classes = {0:'cat', 1:'dog'}
def display_image_grid(images_filepaths, predicted_labels=(), cols=5, save_path='test_dataset.png'):
    rows = len(images_filepaths) // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols,figsize=(12,6))
    for i, image_filepath in enumerate(images_filepaths):
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        a = random.choice(res['id'].values)
        label = res.loc[res['id'] == a, 'label'].values[0]
        if label>0.5:
            label = 1
        else:
            label = 0
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(class_[label])
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path)

display_image_grid(test_files)