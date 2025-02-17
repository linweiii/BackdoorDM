from preact_resnet import PreActResNet18
from resnet import ResNet18
from net_minist import NetC_MNIST
from tqdm import tqdm
from torchvision import datasets
from datasets import load_dataset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import argparse
import os

def train(model, train_loader, criterion, optimizer, scheduler, device, key='image'):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(train_loader, desc="Training", ncols=100):
        inputs = batch[key]
        labels = batch['label']
        # if isinstance(inputs, list):
        #     inputs = torch.tensor(inputs)
        #     print(inputs.shape)
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    scheduler.step()
    return avg_loss, accuracy

def test(model, test_loader, criterion, device, key='image'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", ncols=100):
            inputs = batch[key]
            labels = batch['label']
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
    
    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Train Classifier Models')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cpu or cuda)')
    args = parser.parse_args()
    return args

def add_label_column(ds):
    labels = []
    for idx, record in enumerate(ds):
        heavy_makeup = 1 if record['Heavy_Makeup'] == 1 else 0
        mouth_slightly_open = 1 if record['Mouth_Slightly_Open'] == 1 else 0
        smiling = 1 if record['Smiling'] == 1 else 0
        label = heavy_makeup + 2 * mouth_slightly_open + 4 * smiling
        labels.append(label)
    ds = ds.add_column('label', labels)
    return ds

def get_transforms(dataset, size, train=True):
    trans_list = []
    trans_list.append(transforms.Resize((size, size)))
    if train:
        trans_list.append(transforms.RandomCrop((size, size), padding=5))
        trans_list.append(transforms.RandomRotation(10))
        if dataset == 'CIFAR10':
            trans_list.append(transforms.RandomHorizontalFlip(p=0.5))
    trans_list.append(transforms.ToTensor())
    if dataset == 'CIFAR10':
        trans_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif dataset == 'MNIST':
        trans_list.append(transforms.Normalize([0.5], [0.5]))
    elif dataset == 'CELEBA_ATTR':
        pass
    else:
        raise NotImplementedError()
    return transforms.Compose(trans_list)
        

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    split_method = "train+test"
    key = 'image'
    if args.dataset == 'CIFAR10':
        ds = load_dataset('cifar10', split=split_method)
        trans_train = get_transforms(args.dataset, 32, True)
        trans_test = get_transforms(args.dataset, 32, False)
        def preprocess_data(examples):
            examples['img'] = torch.stack([trans_train(image) for image in examples['img']])
            return examples
        
        def preprocess_data_test(examples):
            examples['img'] = torch.stack([trans_test(image) for image in examples['img']])
            return examples
        
        # ds = ds.with_transform(preprocess_data)
        # print(ds[0])
        
        ds = ds.train_test_split(test_size=int(0.2 * len(ds)))
        train_data = ds['train']
        test_data = ds['test']
        train_data = test_data.with_transform(preprocess_data)
        test_data = train_data.with_transform(preprocess_data_test)
        model = PreActResNet18(num_classes=10).to(device)
        key = 'img'
        ckpt = 'preact_resnet18_cifar10.pth'
    elif args.dataset == 'CELEBA_ATTR':
        ds = load_dataset("tpremoli/CelebA-attrs", split=split_method)
        trans_train = get_transforms(args.dataset, 64, True)
        trans_test = get_transforms(args.dataset, 64, False)
        def preprocess_data(examples):
            examples['image'] = torch.stack([trans_train(image) for image in examples['image']])
            return examples
        
        def preprocess_data_test(examples):
            examples['image'] = torch.stack([trans_test(image) for image in examples['image']])
            return examples
        
        ds = ds.train_test_split(test_size=int(0.2 * len(ds)))
        train_data = add_label_column(ds['train'])
        test_data = add_label_column(ds['test'])
        train_data = train_data.with_transform(preprocess_data)
        test_data = test_data.with_transform(preprocess_data_test)
        model = ResNet18().to(device)
        ckpt = 'resnet18_celeba.pth'
    elif args.dataset == 'MNIST':
        ds = load_dataset("mnist", split=split_method)
        ds = ds.train_test_split(test_size=int(0.2 * len(ds)))
        train_data = ds['train']
        test_data = ds['test']
        trans_train = get_transforms(args.dataset, 28, True)
        trans_test = get_transforms(args.dataset, 28, False)
        
        def preprocess_data(examples):
            examples['image'] = torch.stack([trans_train(image) for image in examples['image']])
            return examples
        
        def preprocess_data_test(examples):
            examples['image'] = torch.stack([trans_test(image) for image in examples['image']])
            return examples
        train_data = train_data.with_transform(preprocess_data)
        test_data = test_data.with_transform(preprocess_data_test)
        model = NetC_MNIST().to(device)
        ckpt = 'net_mnist.pth'
    else:
        raise NotImplementedError()
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 200, 300, 400], 0.1)
    
    for epoch in range(1000):
        print(f"Epoch [{epoch+1}/1000]")
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, scheduler, device, key)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        test_loss, test_acc = test(model, test_loader, criterion, device, key)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    path = os.path.join('classifier_models', ckpt)
    torch.save(model.state_dict, path)
    
if __name__ == "__main__":
    main()