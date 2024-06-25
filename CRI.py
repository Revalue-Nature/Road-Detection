#Load libraries
import copy
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics import BinaryConfusionMatrix
import torch.optim as optim
from torch.autograd import Variable
import os
import time
import numpy as np
import cv2
import rasterio
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from urllib.error import HTTPError
from urllib.request import urlopen
#checking for device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if str(device) == "cuda":
    print("The execution will be on the GPU!")
else:
    print("The execution will be on the CPU!")

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "/Users/minajafari/PycharmProjects/Baseline_segmentation/"

MODELS_PATH = "./models/CRI/"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception, ViT, convnext, swin]
model_name = "ViT"

# Batch size for training (change depending on how much memory you have)
batch_size = 25

# Number of epochs to train for
num_epochs = 3

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False
lr=0.0001
momentum=0.9
num_classes = 1
# Patch size
tam = 16
# Patch shift
stride = 8
OUTPUT_PATH = "./output/PRE/"+model_name

path_images_png = os.path.join(data_dir, 'image_tif')
path_mask_png = os.path.join(data_dir, 'mask_tif')

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

    # Read the mask using rasterio and convert to 'L' mode (grayscale)
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
            croppedMask = mask.crop((x_init, y_init, x_init + tam, y_init + tam))
            croppedIm = im[y_init:y_init + tam, x_init:x_init + tam]
            np_img = np.array(croppedMask)
            np_non_zero = np.count_nonzero(np_img)

            if np_non_zero == 0:
                image_data = {
                    "label": 0,
                    "image": croppedIm,
                    "name_image": filename,
                    "coordinate": (x_init, y_init)
                }
            else:
                image_data = {
                    "label": 1,
                    "image": croppedIm,
                    "name_image": filename,
                    "coordinate": (x_init, y_init)
                }
            dataset_images_crop.append(image_data)
            cont_part += 1
            y_init += stride
        x_init += stride


class RoadClassificationDataset(Dataset):
    def __init__(self, list_images, transform=None):
        self.list_images = list_images
        self.transform = transform

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.list_images[idx]["image"]
        label = torch.tensor(self.list_images[idx]["label"])

        if self.transform:
            image = self.transform(image)

        return image, label

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
    "nicfi_70.tif",  "nicfi_29.tif", "nicfi_63.tif"
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


train_length = len(dataset_images["train"])
val_length = len(dataset_images["val"])

print(f"Number of images in the training set: {train_length}")
print(f"Number of images in the validation set: {val_length}")

print("Number of samples in the class without roads: ")
print(len([im for im in dataset_images_crop if im['label'] == 0]))
print("Number of samples in the class with roads: ")
print(len([im for im in dataset_images_crop if im['label'] == 1]))


def calculate_metrics(TP, TN, FP, FN):
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + FN + FP + TN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    baccuracy = (sensitivity + specificity) / 2
    F1_score = 2 * ((precision * recall) / (precision + recall))
    metrics = {'f1_score': F1_score, 'accuracy': accuracy, 'balanced_accuracy': baccuracy, 'precision': precision,
               'recall': recall}
    return metrics


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            metric = BinaryConfusionMatrix()
            metric.to(device)

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels_old = labels
                labels_old = labels_old.to(device)
                labels = labels.unsqueeze(1).float()
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)

                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                metric.update(outputs.sigmoid().squeeze(1), labels_old)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            mat_conf = metric.compute()
            mat_conf_np = mat_conf.cpu().numpy()
            tp_temp, fn_temp, fp_temp, tn_temp = mat_conf_np.ravel()

            metrics_calculed = calculate_metrics(tp_temp, tn_temp, fp_temp, fn_temp)

            print(
                '{} Loss: {:.4f} Acc_Balanced: {:.4f}'.format(phase, epoch_loss, metrics_calculed["balanced_accuracy"]))

            print(phase + "_epoch_loss", float(epoch_loss))
            for key in metrics_calculed:
                print(phase + "_" + key, metrics_calculed[key])

            # deep copy the model
            if phase == 'val' and metrics_calculed["balanced_accuracy"] > best_acc:
                best_acc = metrics_calculed["balanced_accuracy"]
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(MODELS_PATH, model_name + "-" + 'best_checkpoint.model'))
            if phase == 'val':
                val_acc_history.append(metrics_calculed["balanced_accuracy"])

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(weights=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(weights=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(weights=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(weights=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(weights=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "ViT":
        """ ViT
        """
        model_ft = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        print(model_ft)
        set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.fc.in_features
        model_ft.heads.head = nn.Linear(768, num_classes)
        input_size = 224

    elif model_name == "convnext":
        """ convnext
        """
        model_ft = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[2].in_features
        model_ft.classifier[2] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "swin":
        """ swin
        """
        model_ft = models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1)
        print(model_ft)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.head = nn.Linear(1024, num_classes)
        input_size = 256

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(weights=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(1024, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)

class RemoveExtraChannel(object):
    def __call__(self, img):
        # Assuming the extra channel is the last one, we slice it out
        img = img[:, :, :3]
        return img

data_transforms = {
    'train': transforms.Compose([
        RemoveExtraChannel(),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(0, 359)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        RemoveExtraChannel(),
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        #transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: RoadClassificationDataset(dataset_images[x], data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum=momentum)


if not os.path.isdir(MODELS_PATH):
    os.makedirs (MODELS_PATH)

# Setup the loss fxn
criterion = nn.BCEWithLogitsLoss()

# Train and evaluate
print("data_dir", data_dir)
print("model_name", model_name)
print("batch_size", batch_size)
print("num_epochs", num_epochs)
print("learning_rate", lr)
print("momentum", momentum)

model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

loader = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def image_loader(image):
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  # This is for VGG, may not be needed for ResNet

    if torch.cuda.is_available():
        return image.cuda()  # Assumes that you're using GPU
    else:
        return image  # Use CPU if GPU is not available

def predict_patch(model_ft, image):
  image = image_loader(image)
  model_ft.eval()
  output = model_ft(image)
  sm = torch.nn.Sigmoid()
  probabilities = sm(output)
  result = probabilities.cpu().detach().numpy()

  return result[0]

def toImgOpenCV(imgPIL): # Conver imgPIL to imgOpenCV
    i = np.array(imgPIL) # After mapping from PIL to numpy : [R,G,B,A]
                         # numpy Image Channel system: [B,G,R,A]
    red = i[:,:,0].copy(); i[:,:,0] = i[:,:,2].copy(); i[:,:,2] = red;
    return i;


def toImgPIL(imgOpenCV):
    # Ensure the image data is in the range [0, 255]
    imgOpenCV = np.clip(imgOpenCV, 0, 255)
    # Convert the image data type to uint8
    imgOpenCV = imgOpenCV.astype(np.uint8)
    # Convert the color from BGR to RGB and then create a PIL Image
    return Image.fromarray(cv2.cvtColor(imgOpenCV, cv2.COLOR_BGR2RGB))


if not os.path.isdir(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    os.makedirs(os.path.join(OUTPUT_PATH, "mask"))
    os.makedirs(os.path.join(OUTPUT_PATH, "heatmap"))
    os.makedirs(os.path.join(OUTPUT_PATH, "heatmap_np"))


print(OUTPUT_PATH)

y_true = []
y_pred = []

def send_display_message(url, buffer):
    try:
        urlopen(url, buffer)
    except HTTPError as e:
        if e.code == 429:  # Too Many Requests
            print("Too many requests. Retrying after a delay...")
            time.sleep(60)  # Delay for 60 seconds
            send_display_message(url, buffer)
        else:
            raise

for image_filename in test_filenames:
    print(image_filename)
    x_init = y_init = 0


    path_filename = os.path.join(data_dir, os.path.join("image_tif", image_filename))
    path_filename_mask = os.path.join(data_dir, os.path.join("mask_tif", image_filename))
    with rasterio.open(path_filename) as src:
        image_rast = src.read()  # Read all bands

    im = np.transpose(image_rast, (1, 2, 0))
    im = im[:, :, :3]
    im = 255 * (im - np.min(im)) / (np.max(im) - np.min(im))
    im = im.astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Read the mask using rasterio and convert to 'L' mode (grayscale)
    with rasterio.open(path_filename_mask) as src:
        mask_ras = src.read(1)  # Read the first band

    # Convert to the same type as PIL Image
    mask = Image.fromarray(mask_ras).convert('L')

    reflect = cv2.copyMakeBorder(im, tam, tam, tam, tam, cv2.BORDER_REFLECT)
    im = Image.fromarray(reflect)
    mask = np.array(mask)
    reflect = cv2.copyMakeBorder(mask, tam, tam, tam, tam, cv2.BORDER_REFLECT)
    mask = Image.fromarray(reflect)

    width, height = im.size

    heatmap = np.zeros([height, width], dtype=np.float32)

    while x_init + tam <= width:
        y_init = 0
        while y_init + tam <= height:
            im_cropped = im.crop((x_init, y_init, x_init + tam, y_init + tam))

            croppedMask = mask.crop((x_init, y_init, x_init + tam, y_init + tam))
            np_img = np.array(croppedMask)
            np_non_zero = np.count_nonzero(np_img)

            if np_non_zero == 0:
                y_true.append(0)
            else:
                y_true.append(1)

            result = predict_patch(model_ft, im_cropped)

            if result < 0.5:
                y_pred.append(0)
            else:
                y_pred.append(1)

            heatmap[y_init:y_init + tam, x_init:x_init + tam] += result

            y_init += stride
        x_init += stride

    heatmap = heatmap[tam:height - tam, tam:width - tam]
    plt.imshow(heatmap, cmap='inferno')
    plt.show()

    imagea = (heatmap / np.amax(heatmap))
    imagea = np.uint8(imagea * 255)

    image_mask = imagea
    image_mask[imagea >= 128] = 255
    image_mask[imagea < 128] = 0
    image_mask = Image.fromarray(image_mask, 'L')
    image_mask.save(os.path.join(OUTPUT_PATH, os.path.join("mask", image_filename)))

    img = Image.fromarray(imagea, 'L')
    img_heatmap = np.uint8(img)

    img_heatmap = cv2.applyColorMap(img_heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(OUTPUT_PATH, os.path.join("heatmap_np", image_filename)), img_heatmap)

    with rasterio.open(path_filename) as src:
        image_rast = src.read()  # Read all bands
        im = np.transpose(image_rast, (1, 2, 0))
        im = im[:, :, :3]
        im = 255 * (im - np.min(im)) / (np.max(im) - np.min(im))
        im = im.astype(np.uint8)
        img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


    if img_heatmap.dtype != img.dtype:
        img_heatmap = img_heatmap.astype(img.dtype)

    if img.dtype == np.uint8:
        img_heatmap = np.clip(img_heatmap, 0, 255)
        img = np.clip(img, 0, 255)

    if img.dtype == np.float32:
        max_val = max(img_heatmap.max(), img.max())
        img_heatmap = img_heatmap / max_val
        img = img / max_val
        img_heatmap = np.clip(img_heatmap, 0.0, 1.0)
        img = np.clip(img, 0.0, 1.0)
    elif img.dtype == np.uint8:
        img_heatmap = np.clip(img_heatmap, 0, 255)
        img = np.clip(img, 0, 255)
    super_imposed_img = cv2.addWeighted(img_heatmap, 0.6, img, 1, 0)
    cv2.imwrite(os.path.join(OUTPUT_PATH, os.path.join("heatmap", image_filename)), super_imposed_img)
    plt.imshow(super_imposed_img)
    plt.show()

print(classification_report(y_true, y_pred))


