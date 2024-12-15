import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import smdebug.pytorch as smd
from smdebug.pytorch import Hook
import argparse
from urllib.parse import urlparse

class MyImageDataset(Dataset):
    def __init__(self, data_dir, labels_file, transform=None):
        self.data_dir = data_dir
        self.labels_file = labels_file
        self.label_map = self._load_labels_from_file()
        self.image_paths = self._get_image_paths()
        self.transform = transform if transform else transforms.ToTensor()

    def _get_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith('.png'):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def _load_labels_from_file(self):
        label_map = {}
        with open(self.labels_file, 'r') as f:
            for line in f:
                filename, label = line.strip().split(',')
                label_map[filename] = int(label)
        return label_map

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        filename = os.path.basename(image_path)
        label = self.label_map.get(filename, -1)
        return image, label


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    validation_accuracy = 100 * correct / total
    print(f'Validation Accuracy: {validation_accuracy:.4f}')


def train(model, train_loader, criterion, optimizer, device, epochs):
    epochs = int(epochs)
    model.train()
    total_processed_images = 0
    processed_images = 0
    for e in range(epochs):
        running_loss = 0
        correct = 0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            pred = pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed_images += 1
            total_processed_images += 1
            if processed_images >= 200:
                print(f"Processed {total_processed_images} images - Epoch {e}: Loss {running_loss / len(train_loader)}, Accuracy {100 * correct/len(train_loader.dataset)}%")
                processed_images = 0 
        print(f"Epoch {e}: Loss {running_loss / len(train_loader)}, Accuracy {100 * correct / len(train_loader.dataset)}%")


def net():
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 100))
    return model


def create_data_loaders(data, batch_size):
    loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs

    model = net().to(device)
    
    train_dataset = MyImageDataset("/opt/ml/input/data/train", "/opt/ml/input/data/trainlabel/train.txt")
    print("create train_dataset done")
    test_dataset = MyImageDataset("/opt/ml/input/data/test", "/opt/ml/input/data/testlabel/test.txt")
    print("create test_dataset done")
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print("create train_loader done")
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print("create test_loader done")

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

    train(model, train_loader, loss_criterion, optimizer, device, epochs)
    test(model, test_loader, device)

    torch.save(model.state_dict(), "/opt/ml/model/model.pth")
    
def extract_bucket_and_prefix(s3_path):
    parsed_url = urlparse(s3_path)
    bucket_name = parsed_url.netloc
    prefix = parsed_url.path.lstrip('/')
    return bucket_name, prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    args = parser.parse_args()
    
    main(args)