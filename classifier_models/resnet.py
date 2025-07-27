import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import CocoDetection
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

class CatClassifierDataset(Dataset):
    def __init__(self, root, annFile, transform=None, neg_pos_ratio=1.0):
        self.coco = CocoDetection(root=root, annFile=annFile, transform=None)
        self.transform = transform
        self.neg_pos_ratio = neg_pos_ratio
        
        # Get category ID for cat
        self.cat_id = None
        for cat in self.coco.coco.cats.values():
            if cat['name'] == 'cat':
                self.cat_id = cat['id']
                break
        
        if self.cat_id is None:
            raise ValueError("Could not find cat category in COCO dataset")
        
        # Filter images containing cats and non-cats
        self.cat_indices = []  # Images with cats
        self.non_cat_indices = []  # Images without cats
        
        for idx, (_, anns) in enumerate(tqdm(self.coco, desc="Filtering dataset")):
            has_cat = False
            
            for ann in anns:
                if ann['category_id'] == self.cat_id:
                    has_cat = True
                    break
            
            if has_cat:
                self.cat_indices.append(idx)
            else:
                self.non_cat_indices.append(idx)
        
        # Sample non-cat images based on the ratio
        num_cats = len(self.cat_indices)
        num_non_cats_to_sample = int(num_cats * self.neg_pos_ratio)
        
        # Ensure we don't try to sample more than available
        num_non_cats_to_sample = min(num_non_cats_to_sample, len(self.non_cat_indices))
        
        # Randomly sample non-cat images
        sampled_non_cat_indices = np.random.choice(
            self.non_cat_indices, 
            size=num_non_cats_to_sample, 
            replace=False
        ).tolist()
        
        # Combine cat and sampled non-cat indices
        self.filtered_indices = self.cat_indices + sampled_non_cat_indices
        
        # Create labels: 1 for cat, 0 for non-cat
        self.labels = [1] * len(self.cat_indices) + [0] * len(sampled_non_cat_indices)
        
        print(f"Dataset created with {len(self.cat_indices)} cat images and {len(sampled_non_cat_indices)} non-cat images")
    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        img, _ = self.coco[self.filtered_indices[idx]]
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label


class PatchDataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        """
        Dataset for training a model to detect images with patches.
        
        Args:
            root (string): Directory with COCO dataset images
            annFile (string): Path to COCO annotation file
            transform (callable, optional): Optional transform to be applied on images
        """
        self.coco = CocoDetection(root=root, annFile=annFile, transform=None)
        self.transform = transform
        
        # Sample 10k images randomly
        all_indices = list(range(len(self.coco)))
        self.sampled_indices = np.random.choice(
            all_indices, 
            size=min(10000, len(all_indices)), 
            replace=False
        ).tolist()
        
        # Load the target patch image
        self.patch_img = Image.open('./utils/pixel_target/boya.jpg').convert('RGB')
        self.patch_img = self.patch_img.resize((128, 128), Image.LANCZOS)
        
        # Parameters for patch placement
        self.sit_w = 0
        self.sit_h = 0
        
        # Create pairs of images (original and with patch)
        self.filtered_indices = []
        self.labels = []
        
        # For each sampled image, we'll have two entries:
        # 1. Original image (label 0)
        # 2. Image with patch (label 1)
        for idx in self.sampled_indices:
            # Original image
            self.filtered_indices.append(idx)
            self.labels.append(0)
            
            # Same image with patch
            self.filtered_indices.append(idx)
            self.labels.append(1)
        
        print(f"Dataset created with {len(self.sampled_indices)} original images and {len(self.sampled_indices)} patched images")
    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        img_idx = self.filtered_indices[idx]
        img, _ = self.coco[img_idx]
        label = self.labels[idx]
        
        # Add patch if this is a patched image
        if label == 1:
            img = img.copy()
            img.paste(self.patch_img, (self.sit_w, self.sit_h))
            # img.save('sample_patch.jpg')
            exit()
        
        if self.transform:
            img = self.transform(img)
            
        return img, label



def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device='cuda'):
    best_acc = 0.0
    best_model_weights = None
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        
        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Save the best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_weights = model.state_dict().copy()
    
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    return model

def train(mode):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data paths
    coco_root = "./train2017"
    coco_annFile = "annotations/instances_train2017.json"
    
    # Data transforms
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    print("Loading and filtering COCO dataset...")
    if mode == 'cat':
        dataset = CatClassifierDataset(root=coco_root, annFile=coco_annFile, transform=data_transforms)
    elif mode == 'patch':
        dataset = PatchDataset(root=coco_root, annFile=coco_annFile, transform=data_transforms)
   
    # Split dataset into train and validation
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Dataset size: {dataset_size}, Train size: {train_size}, Val size: {val_size}")
    
    # Load pre-trained ResNet50 model
    if mode == 'cat':
        model = models.resnet50(pretrained=True)
    elif mode == 'patch':
        model = models.resnet18(pretrained=True)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Binary classification: dog (0) or cat (1)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train the model
    print("Starting training...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=20,
        device=device
    )
    
    # Save the trained model
    save_dir = './classifiers'
    os.makedirs(save_dir, exist_ok=True)
    if mode == 'cat':
        model_path = f'{save_dir}/resnet50_dog_cat_classifier.pth'
        torch.save(model.state_dict(), model_path)
    elif mode == 'patch':
        model_path = f'{save_dir}/resnet18_patch_classifier.pth'
        torch.save(model.state_dict(), model_path)

    print("Training complete. Model saved.")
    return model_path


def test_model(model_path, test_dir, mode, device='cuda'):
    """
    Test the trained model on a test dataset
    
    Args:
        model_path: Path to the saved model
        test_dir: Directory containing test images
        device: Device to run the model on
    
    Returns:
        accuracy: Test accuracy
    """
    # Load the model
    if mode == 'cat':
        model = models.resnet50(pretrained=False)
    elif mode == 'patch':
        model = models.resnet18(pretrained=False)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Binary classification: dog (0) or cat (1)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Define transforms for test data
    test_transforms = transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load all images from the directory
    image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images in {test_dir}")
    
    # Test the model
    correct = 0
    total = len(image_files)
    
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Testing images"):
            try:
                # Load and preprocess the image
                img = Image.open(img_path).convert('RGB')
                img_tensor = test_transforms(img).unsqueeze(0).to(device)
                
                # Make prediction
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                
                # If predicted as cat (class 1), count as correct
                if predicted.item() == 1:  # 1 is the cat class
                    correct += 1
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                total -= 1  # Don't count this image
    
    if total > 0:
        accuracy = 100 * correct / total
        print(f'{mode} prediction rate: {accuracy:.2f}% ({correct}/{total} images predicted as {mode})')
    else:
        accuracy = 0
        print("No valid images were processed")
    
    return accuracy


if __name__ == "__main__":
    is_train = False
    is_test = True
    mode = 'patch'

    if is_train:
        model_path = train(mode)
    if is_test:
        test_dir = "test_patch"
        model_path = './classifiers/resnet18_patch_classifier.pth'
        test_model(model_path=model_path, test_dir=test_dir, mode=mode, device='cuda:0')
