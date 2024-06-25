import os
import cv2
from PIL import Image
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, LovaszLoss,JaccardLoss, FocalLoss
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import albumentations as A

DATA_DIR = "/Users/minajafari/PycharmProjects/Baseline_segmentation/"
MODELS_PATH = "./models/PRE/"
OUTPUT_DIR = "./output/PRE/"
# if torch.backends.mps.is_available():
#     DEVICE = torch.device("mps")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


EPOCHS=7
LR=0.0003
IMAGE_SIZE= 128
BATCH_SIZE= 44

# Encoder to choose from ["timm-efficientnet-b0", "timm-efficientnet-b7", "resnext101_32x8d"]
ENCODER = 'timm-efficientnet-b7'
# Models to choose from ["DeepLabV3Plus","UnetPlusPlus","Unet"]
MODEL = "UnetPlusPlus"
# Weights to choose from ["imagenet","noisy-student"]
WEIGHTS = 'imagenet'

# Patch size
tam = 128
# Patch shift
stride = 64

def get_train_augs():
  return A.Compose([
      A.Resize(IMAGE_SIZE,IMAGE_SIZE),
      A.HorizontalFlip(p=.5),
      A.VerticalFlip(p=.5),
      A.Rotate()
  ])
def get_valid_augs():
    return A.Compose([
      A.Resize(IMAGE_SIZE,IMAGE_SIZE)
  ])

path_images_png = os.path.join(DATA_DIR, 'image_tif')
path_mask_png = os.path.join(DATA_DIR, 'mask_tif')

images_filenames = os.listdir(path_images_png)
images_mask_filenames = os.listdir(path_mask_png)

if len(images_filenames) != len(images_mask_filenames):
    print("Number of images does not correspond to the number of masks!")

dataset_images_crop = []

for filename in images_filenames:
    with rasterio.open(os.path.join(path_images_png, filename)) as src:
        image_rast = src.read()  # Read all bands

    im = np.transpose(image_rast, (1, 2, 0))
    im = im[:, :, :3]
    im = 255 * (im - np.min(im)) / (np.max(im) - np.min(im))
    im = im.astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    with rasterio.open(os.path.join(path_mask_png, filename)) as src:
        mask_ras = src.read(1)  # Read the first band

    mask = Image.fromarray(mask_ras).convert('L')

    width, height = mask.size
    x_init = y_init = 0
    cont_part = 0

    print(filename)

    while x_init + tam <= width:
        y_init = 0
        while y_init + tam <= height:
            patch_image = im[y_init:y_init + tam, x_init:x_init + tam]
            patch_mask_ground_truth = mask.crop((x_init, y_init, x_init + tam, y_init + tam))

            np_img = np.array(patch_mask_ground_truth)
            np_non_zero = np.count_nonzero(np_img)

            if np_non_zero == 0:
                image_data = {
                    "label": 0,
                    "mask": np_img,
                    "image": patch_image,
                    "name_image": filename,
                    "coordinate": (x_init, y_init)
                }
            else:
                image_data = {
                    "label": 1,
                    "mask": np_img,
                    "image": patch_image,
                    "name_image": filename,
                    "coordinate": (x_init, y_init)
                }
            dataset_images_crop.append(image_data)
            cont_part += 1
            y_init += stride
        x_init += stride


class SegmentationDataset(Dataset):
    def __init__(self, list_images, augmentations):
        self.list_images = list_images
        self.augmentations = augmentations

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.list_images[idx]["image"]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = self.list_images[idx]["mask"]
        mask = np.expand_dims(mask, axis=-1)

        if self.augmentations:
            data = self.augmentations(image=image, mask=mask)
            image = data['image']
            mask = data['mask']

            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

            image = torch.Tensor(image) / 255.0
            mask = torch.round(torch.Tensor(mask) / 255.0)

            return image, mask


def show_image(image, mask, pred_image=None):
    if pred_image == None:

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.set_title('IMAGE')
        ax1.imshow(image.permute(1, 2, 0).squeeze(), cmap='gray')

        ax2.set_title('GROUND TRUTH')
        ax2.imshow(mask.permute(1, 2, 0).squeeze(), cmap='gray')

    elif pred_image != None:

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

        ax1.set_title('IMAGE')
        ax1.imshow(image.permute(1, 2, 0).squeeze(), cmap='gray')

        ax2.set_title('GROUND TRUTH')
        ax2.imshow(mask.permute(1, 2, 0).squeeze(), cmap='gray')

        ax3.set_title('MODEL OUTPUT')
        ax3.imshow(pred_image.permute(1, 2, 0).squeeze(), cmap='gray')

train_filenames = [
    "nicfi_109.tif", "nicfi_1.tif", "nicfi_113.tif", "nicfi_101.tif", "nicfi_37.tif",
    "nicfi_123.tif",
    "nicfi_20.tif", "nicfi_98.tif",  "nicfi_12.tif",
    "nicfi_2.tif",  "nicfi_94.tif", "nicfi_73.tif",
    "nicfi_108.tif", "nicfi_92.tif", "nicfi_25.tif",
    "nicfi_102.tif", "nicfi_72.tif", "nicfi_64.tif",
    "nicfi_24.tif", "nicfi_3.tif", "nicfi_87.tif",
    "nicfi_114.tif", "nicfi_34.tif", "nicfi_17.tif",
    "nicfi_122.tif", "nicfi_49.tif", "nicfi_28.tif",
    "nicfi_121.tif", "nicfi_116.tif",
    "nicfi_105.tif", "nicfi_117.tif", "nicfi_5.tif",
    "nicfi_104.tif", "nicfi_124.tif", "nicfi_115.tif", "nicfi_62.tif",
    "nicfi_19.tif", "nicfi_42.tif",
    "nicfi_40.tif", "nicfi_7.tif", "nicfi_35.tif", "nicfi_69.tif", "nicfi_81.tif",
    "nicfi_27.tif", "nicfi_13.tif",
    "nicfi_9.tif", "nicfi_78.tif", "nicfi_110.tif", "nicfi_30.tif", "nicfi_41.tif",
    "nicfi_57.tif", "nicfi_8.tif", "nicfi_86.tif", "nicfi_56.tif", "nicfi_44.tif",
    "nicfi_85.tif", "nicfi_45.tif", "nicfi_4.tif",  "nicfi_18.tif",
    "nicfi_75.tif",
    "nicfi_70.tif",  "nicfi_29.tif", "nicfi_63.tif",

]

test_filenames = [
    "nicfi_88.tif", "nicfi_100.tif", "nicfi_71.tif",
    "nicfi_82.tif", "nicfi_67.tif", "nicfi_107.tif",
    "nicfi_48.tif", "nicfi_10.tif", "nicfi_6.tif",
     "nicfi_112.tif", "nicfi_120.tif", "nicfi_119.tif", "nicfi_103.tif",
    "nicfi_118.tif", "nicfi_106.tif"
]


dataset_images = {}
dataset_images["train"] = [im for im in dataset_images_crop if im['name_image'] in train_filenames]
dataset_images["val"] = [im for im in dataset_images_crop if im['name_image'] in test_filenames]

trainset = SegmentationDataset(dataset_images["train"], get_train_augs())
validset = SegmentationDataset(dataset_images["val"], get_valid_augs())

print(f"Size of Trainset : {len(trainset)}")
print(f"Size of Validset : {len(validset)}")

idx = 30

print(trainset)

image, mask = trainset[idx]
show_image(image,mask)

trainloader = DataLoader(trainset,batch_size = BATCH_SIZE, shuffle = True)
validloader = DataLoader(validset, batch_size = BATCH_SIZE)

print(f"Total number of batches in train loader:  {len(trainloader)}")
print(f"Total number of batches in valid loader:  {len(validloader)}")

for image,mask in trainloader:
  break

print(f"One Batch Image Shape: {image.shape}")
print(f"One Batch Mask Shape: {mask.shape}")

import torch
from torch import nn
from torch.nn import functional as F

class CustomFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(CustomFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(inputs.dtype)  # Ensure same data type
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# Use the custom loss function in your training function
loss_fn = CustomFocalLoss()


class SegmentationModel(nn.Module):

  def __init__(self, model = "DeepLabV3Plus"):
    super(SegmentationModel, self).__init__()

    if model == "DeepLabV3Plus":
        self.arc = smp.DeepLabV3Plus(
            encoder_name = ENCODER,
            encoder_weights= WEIGHTS,
            in_channels = 3,
            classes = 1,
            activation = None
        )
    if model == "Unet":
        self.arc = smp.Unet(
            encoder_name = ENCODER,
            encoder_weights= WEIGHTS,
            in_channels = 3,
            classes = 1,
            activation = None
        )
    if model == "UnetPlusPlus":
        self.arc = smp.UnetPlusPlus(
            encoder_name = ENCODER,
            encoder_weights= WEIGHTS,
            in_channels = 3,
            classes = 1,
            activation = None
        )

  def forward(self,images,masks=None):
    logits = self.arc(images)

    if masks != None:

      loss1 = JaccardLoss(mode='binary')(logits,masks)
      logits = logits.to('mps').float()
      masks = masks.to('mps').float()
      loss2 = loss_fn(logits,masks)
      return logits,loss1+loss2#+loss3
    return logits

model = SegmentationModel(MODEL)
model.to(DEVICE)


def calc_metrics(tp, fp, fn, tn):
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
    balanced_accuracy = smp.metrics.balanced_accuracy(tp, fp, fn, tn, reduction="micro")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
    precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
    metrics = {'iou': iou_score, 'f1_score': f1_score, 'f2_score': f2_score, 'accuracy': accuracy,
               'balanced_accuracy': balanced_accuracy, 'recall': recall, 'precision': precision}

    return metrics


def train_fn(data_loader, model, optimizer):
    model.train()
    total_loss = 0

    tp, fp, fn, tn = None, None, None, None

    for images, masks in tqdm(data_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        logits, loss = model(images, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        labels = logits

        tp_temp, fp_temp, fn_temp, tn_temp = smp.metrics.get_stats(labels.to(DEVICE), masks.long().to(DEVICE),
                                                                   mode='binary', threshold=0.5)

        if tp == None:
            tp = tp_temp
            fp = fp_temp
            fn = fn_temp
            tn = tn_temp
        else:
            tp = torch.cat([tp, tp_temp])
            fp = torch.cat([fp, fp_temp])
            fn = torch.cat([fn, fn_temp])
            tn = torch.cat([tn, tn_temp])

    d1 = calc_metrics(tp, fp, fn, tn)

    return total_loss / len(data_loader), d1


def eval_fn(data_loader, model):
    model.eval()
    total_loss = 0
    with torch.no_grad():

        tp, fp, fn, tn = None, None, None, None

        for images, masks in tqdm(data_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            logits, loss = model(images, masks)
            total_loss += loss.item()
            labels = logits

            tp_temp, fp_temp, fn_temp, tn_temp = smp.metrics.get_stats(labels.to(DEVICE), masks.long().to(DEVICE),
                                                                       mode='binary', threshold=0.5)

            if tp == None:
                tp = tp_temp
                fp = fp_temp
                fn = fn_temp
                tn = tn_temp
            else:
                tp = torch.cat([tp, tp_temp])
                fp = torch.cat([fp, fp_temp])
                fn = torch.cat([fn, fn_temp])
                tn = torch.cat([tn, tn_temp])

    d1 = calc_metrics(tp, fp, fn, tn)

    return total_loss / len(data_loader), d1

optimizer = torch.optim.Adam(model.parameters(), lr = LR)
best_iou = 0

if not os.path.isdir(MODELS_PATH):
    os.makedirs(MODELS_PATH)

print("EPOCHS", EPOCHS)
print("LR", LR)
print("IMAGE_SIZE", IMAGE_SIZE)
print("BATCH_SIZE", BATCH_SIZE)
print("ENCODER", ENCODER)
print("MODEL", MODEL)
print("WEIGHTS", WEIGHTS)

for i in range(EPOCHS):
    train_loss, train_metrics = train_fn(trainloader, model, optimizer)
    valid_loss, valid_metrics = eval_fn(validloader, model)

    if valid_metrics["iou"].item() > best_iou:
        torch.save(model.state_dict(), os.path.join(MODELS_PATH, MODEL + "-" + ENCODER + "-" + 'best_model.pt'))
        print("Saved Model")
        best_iou = valid_metrics["iou"].item()

    print(f"Epoch = {i + 1} Train loss : {train_loss} Valid loss {valid_loss}")

    print("TRAIN_LOSS", train_loss)
    print("VALID_LOSS", valid_loss)

    for key in train_metrics:
        print("TRAIN_" + key, train_metrics[key].item())

    for key in valid_metrics:
        print(key, ": ", valid_metrics[key].item())
        print("VALID_" + key, valid_metrics[key].item())

torch.save(model.state_dict(), os.path.join(MODELS_PATH, MODEL + "-" + ENCODER + "-" + 'final_model.pt'))
print("Saved Final Model")

print(os.path.join(MODELS_PATH, MODEL + "-" + ENCODER + "-" + "best_model.pt"))
print(os.path.join(MODELS_PATH, MODEL + "-" + ENCODER + "-" + "final_model.pt"))


idx = 30

model.load_state_dict(torch.load(os.path.join(MODELS_PATH, MODEL+"-"+ENCODER+"-"+'final_model.pt')))
model.eval()
image, mask = validset[idx]
logits_mask = model(image.to(DEVICE).unsqueeze(0))
pred_mask = torch.sigmoid(logits_mask)
show_image(image,mask,pred_mask.detach().cpu().squeeze(0))

def predict_patch(model_ft, image):
    model.eval()
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.Tensor(image) / 255.0
    logits_mask = model(image.to(DEVICE).unsqueeze(0))
    pred_mask = torch.sigmoid(logits_mask)

    return pred_mask

output_path_all_patches = os.path.join(OUTPUT_DIR, MODEL+"-"+ENCODER+"-"+WEIGHTS+"-"+"all_patches")
if not os.path.isdir(output_path_all_patches):
    os.makedirs (output_path_all_patches)

for image_filename in test_filenames:
    with rasterio.open(os.path.join(path_images_png, image_filename)) as src:
        image_rast = src.read()  # Read all bands
    im = np.transpose(image_rast, (1, 2, 0))
    im = im[:, :, :3]
    im = 255 * (im - np.min(im)) / (np.max(im) - np.min(im))
    im = im.astype(np.uint8)
    image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    with rasterio.open(os.path.join(path_mask_png, image_filename)) as src:
        mask_ras = src.read(1)  # Read the first band

    mask = Image.fromarray(mask_ras).convert('L')

    x_init = y_init = 0
    old_image_height, old_image_width, channels = image.shape
    image_padding = cv2.copyMakeBorder(image.copy(), tam, tam, tam, tam, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    image_height, image_width, channels = image_padding.shape
    segmentation = np.zeros([image_height, image_width], dtype=np.float32)

    while x_init + tam <= image_width:
        y_init = 0
        while y_init + tam <= image_height:
            image_eval = image_padding[y_init:y_init + tam, x_init:x_init + tam]
            pred_mask = predict_patch(model, image_eval)
            segmentation[y_init:y_init + tam, x_init:x_init + tam] += pred_mask.cpu().detach().numpy()[0][0]
            y_init += stride
        x_init += stride

    segmentation = segmentation[tam: image_height - tam, tam: image_width - tam]
    # print(segmentation.shape)
    # plt.imshow(image)
    # plt.show()
    # plt.imshow(segmentation, cmap='inferno')
    # plt.show()

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

    ax1.set_title('IMAGE')
    ax1.imshow(image)

    ax2.set_title('GROUND TRUTH')
    ax2.imshow(mask)

    ax3.set_title('MODEL OUTPUT')
    ax3.imshow(segmentation)
    plt.show()

    mask_gray = cv2.normalize(src=segmentation, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_8UC1)

    mask_gray[mask_gray >= 70] = 255
    mask_gray[mask_gray < 70] = 0

    # Automatically calculate the threshold value using Otsu's method
    # _, mask_gray = cv2.threshold(mask_gray_, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 'mask_binarized' is the binary image obtained using the Otsu's automatically determined threshold

    cv2.imwrite(os.path.join(output_path_all_patches, image_filename), mask_gray)


def calculate_metrics(TP, TN, FP, FN):
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + FN + FP + TN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    baccuracy = (sensitivity + specificity) / 2
    IOU = TP / (TP + FP + FN)
    F1_score = 2 * ((precision * recall) / (precision + recall))
    metrics = {'iou': IOU, 'f1_score': F1_score, 'accuracy': accuracy, 'balanced_accuracy': baccuracy,
               'precision': precision, 'recall': recall}
    return metrics

groundtruth_path = os.path.join(DATA_DIR, "mask_tif")
tp = tn = fp = fn = 0

for filename in os.listdir(output_path_all_patches):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        with rasterio.open(os.path.join(output_path_all_patches, filename)) as src:
            mask_ras = src.read(1)  # Read the first band
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        with rasterio.open(os.path.join(groundtruth_path, filename)) as src:
            gt_ras = src.read(1)  # Read the first band

    mask = Image.fromarray(mask_ras).convert('L')
    # Convert to the same type as PIL Image
    groundtruth = Image.fromarray(gt_ras).convert('L')

    mask_array = np.array(mask).flatten()
    groundtruth_array = np.array(groundtruth).flatten()
    tn_temp, fp_temp, fn_temp, tp_temp = confusion_matrix(groundtruth_array, mask_array).ravel()

    tp += tp_temp
    tn += tn_temp
    fp += fp_temp
    fn += fn_temp

metrics = calculate_metrics(tp, tn, fp, fn)

filter_string = "attributes.run_name ILIKE '%" + ENCODER + "_" + MODEL + "%'"
for key in metrics:
    print(f"{key} : {metrics[key]}")
    print("TEST_" + key, metrics[key])