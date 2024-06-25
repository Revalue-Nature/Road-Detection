import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms
from sklearn.model_selection import train_test_split
import random
import os
import numpy as np
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
def calculate_pos_weight(mask_dir):
    total_pixels = 0
    road_pixels = 0

    for mask_name in os.listdir(mask_dir):
        if mask_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            mask_path = os.path.join(mask_dir, mask_name)
            mask = np.array(Image.open(mask_path).convert("L"))  # Convert mask to grayscale
            total_pixels += mask.size
            road_pixels += np.sum(mask > 0)  # Assuming road pixels are non-zero in the mask

    # Calculate the ratio
    background_pixels = total_pixels - road_pixels
    pos_weight = background_pixels / road_pixels
    return pos_weight


# Calculate pos_weight based on your dataset
mask_dir = '/Users/minajafari/PycharmProjects/Baseline_segmentation/lab'
pos_weight = calculate_pos_weight(mask_dir)
print(f'Calculated pos_weight: {pos_weight}')


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.match_dimensions = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.match_dimensions = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # print(f'Output shape: {out.shape}, Residual shape: {residual.shape}')

        if self.match_dimensions is not None:
            residual = self.match_dimensions(residual)
        # print(f'Output shape: {out.shape}, Residual shape: {residual.shape}')

        # Explicitly pad the residual if its size doesn't match 'out'
        if out.shape[2:] != residual.shape[2:]:
            height_pad = (out.shape[2] - residual.shape[2]) // 2
            width_pad = (out.shape[3] - residual.shape[3]) // 2
            residual = F.pad(residual, [width_pad, width_pad + (out.shape[3] - residual.shape[3] - 2 * width_pad),
                                        height_pad, height_pad + (out.shape[2] - residual.shape[2] - 2 * height_pad)])
        # print(f'Output shape: {out.shape}, Residual shape: {residual.shape}')

        out += residual
        out = self.relu(out)
        return out



class ResUNet(nn.Module):
    def __init__(self):
        super(ResUNet, self).__init__()
        # Define Encoder, Bottleneck, and Decoder
        self.encoder = nn.Sequential(
            ResBlock(4, 64),
            nn.MaxPool2d(2),
            ResBlock(64, 128),
            nn.MaxPool2d(2),
            ResBlock(128, 256),
            nn.MaxPool2d(2),
        )
        self.bottleneck = ResBlock(256, 512)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            ResBlock(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            ResBlock(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            ResBlock(64, 64)
        )
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        return x


def init_weights(m):
    if type(m) == nn.Conv2d:
        if m.in_channels == 4:
            # Initialize the first three channels with pretrained weights and the last channel randomly
            pretrained_weights = torch.load('/Users/minajafari/PycharmProjects/Baseline_segmentation/pre-trained_model/imagenet_pretrained.pth')['conv1.weight']
            new_weights = torch.randn_like(pretrained_weights[:, :1, :, :])  # Random weights for the 4th channel
            m.weight.data = torch.cat((pretrained_weights, new_weights), dim=1)
        else:
            nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

model = ResUNet()
model.apply(init_weights)


class MyTransform:
    def __init__(self, size=(512, 512)):  # Define the desired output size here
        self.size = size
        self.transforms = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            # Optionally add normalization or other transformations
        ])

    def __call__(self, image, mask):
        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size)
        # image = TF.to_tensor(image)
        # mask = TF.to_tensor(mask)

        # Apply random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Apply random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Apply random rotation
        angle = random.choice([0, 90, 180, 270])
        if angle > 0:
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # Convert image and mask to tensors
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask





class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform or MyTransform()
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        # mask_path = os.path.join(self.mask_dir, self.images[index].replace(".bmp", "_mask.bmp"))
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("image", "").replace(".png", "_mask.png"))

        image = Image.open(img_path).convert("RGBA")  # Ensure image has 4 channels
        mask = Image.open(mask_path).convert("L")  # Mask assumed to be in grayscale

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


transform = transforms.Compose([
    transforms.ToTensor(),
    # Add other necessary transforms
])

dataset = SegmentationDataset("/Users/minajafari/PycharmProjects/Baseline_segmentation/img", "/Users/minajafari/PycharmProjects/Baseline_segmentation/lab")
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)



import torch.optim as optim
import torch.nn.functional as F

pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)


def dice_coeff(pred, target):
    smooth = 1.
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def calculate_metrics(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return precision.item(), recall.item(), f1.item()



# Split data into train and validation
train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

train_loader = DataLoader(dataset, batch_size=20, sampler=train_subsampler)
val_loader = DataLoader(dataset, batch_size=8, sampler=val_subsampler)


log_interval = 10

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)

        # Resize output to match the target size
        output = F.interpolate(output, size=target.size()[2:], mode='bilinear', align_corners=False)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')



def validate(loader):
    model.eval()
    validation_loss = 0
    dice_score = 0
    all_precision = 0
    all_recall = 0
    all_f1 = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)

            # Resize output to match the target size
            output = F.interpolate(output, size=target.size()[2:], mode='bilinear', align_corners=False)

            loss = criterion(output, target)
            validation_loss += loss.item()

            dice_score += dice_coeff(output, target).item()
            precision, recall, f1 = calculate_metrics(output, target)
            all_precision += precision
            all_recall += recall
            all_f1 += f1

    validation_loss /= len(loader)
    dice_score /= len(loader)
    all_precision /= len(loader)
    all_recall /= len(loader)
    all_f1 /= len(loader)
    print(f'\nValidation set: Average loss: {validation_loss:.4f}, Dice Coeff: {dice_score:.4f}, '
          f'Precision: {all_precision:.4f}, Recall: {all_recall:.4f}, F1 Score: {all_f1:.4f}\n')


for epoch in range(1, 100):  # For example, train for 20 epochs
    train(epoch)
    validate(val_loader)


def save_predictions(loader, folder="predictions"):
    model.eval()
    os.makedirs(folder, exist_ok=True)
    for i, (data, _) in enumerate(loader):
        with torch.no_grad():
            output = model(data)
            output = torch.sigmoid(output)
            output = (output > 0.5).float()
        for j, pred in enumerate(output):
            save_image(pred, os.path.join(folder, f"pred_{i * loader.batch_size + j}.png"))

# Example usage
test_loader = DataLoader(dataset, batch_size=20, sampler=val_subsampler)
save_predictions(test_loader)
