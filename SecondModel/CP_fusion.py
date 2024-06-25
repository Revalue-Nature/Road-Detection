import cv2
import os
import numpy as np
import rasterio
from PIL import Image
from sklearn.metrics import confusion_matrix


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


def log_metrics(input_path, filter_mask_path, groundtruth_path, output_path):
    print(input_path)
    print(filter_mask_path)
    tp = tn = fp = fn = 0
    kernel = np.ones((3, 3), np.uint8)

    for filename in os.listdir(input_path):
        # mask = cv2.imread(os.path.join(input_path, filename), cv2.IMREAD_GRAYSCALE)
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            with rasterio.open(os.path.join(input_path, filename)) as src:
                mask_ras = src.read(1)  # Read the first band

        mask = Image.fromarray(mask_ras).convert('L')
        mask = cv2.dilate(np.array(mask), kernel, iterations=1)
        # groundtruth = cv2.imread(os.path.join(groundtruth_path, filename), cv2.IMREAD_GRAYSCALE)
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            with rasterio.open(os.path.join(groundtruth_path, filename)) as src:
                gt_ras = src.read(1)  # Read the first band

        groundtruth = Image.fromarray(gt_ras).convert('L')
        groundtruth = cv2.dilate(np.array(groundtruth), kernel, iterations=1)

        tn_temp, fp_temp, fn_temp, tp_temp = confusion_matrix(groundtruth.flatten(), mask.flatten()).ravel()

        tp += tp_temp
        tn += tn_temp
        fp += fp_temp
        fn += fn_temp

    metrics = calculate_metrics(tp, tn, fp, fn)
    ################ Apply Filter of Model Classification ################
    tp = tn = fp = fn = 0

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(input_path):
        # mask = cv2.imread(os.path.join(input_path, filename), cv2.IMREAD_GRAYSCALE)
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            with rasterio.open(os.path.join(input_path, filename)) as src:
                mask_ras = src.read(1)  # Read the first band

        mask = Image.fromarray(mask_ras).convert('L')
        mask = cv2.dilate(np.array(mask), kernel, iterations=1)
        # groundtruth = cv2.imread(os.path.join(groundtruth_path, filename), cv2.IMREAD_GRAYSCALE)
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            with rasterio.open(os.path.join(groundtruth_path, filename)) as src:
                gt_ras = src.read(1)  # Read the first band

        groundtruth = Image.fromarray(gt_ras).convert('L')
        groundtruth = cv2.dilate(np.array(groundtruth), kernel, iterations=1)
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            mask_filter = cv2.imread(os.path.join(filter_mask_path, filename), cv2.IMREAD_GRAYSCALE)

        # print(mask.shape)
        # print(mask_filter.shape)
        mask_filtered = cv2.bitwise_and(mask, mask_filter)
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            cv2.imwrite(os.path.join(output_path, filename), mask_filtered)

        c_m = confusion_matrix(groundtruth.flatten(), mask_filtered.flatten(), labels=[0, 255])
        '''if c_m.shape[0] == 1 and c_m.shape[1] == 1:
            tn_temp, fp_temp, fn_temp, tp_temp = c_m.ravel()[0], 0 , 0 , 0
        else:'''
        tn_temp, fp_temp, fn_temp, tp_temp = c_m.ravel()
        # tn_temp, fp_temp, fn_temp, tp_temp = np.float64(tn_temp), np.float64(fp_temp), np.float64(fn_temp), np.float64(tp_temp)

        tp += tp_temp
        tn += tn_temp
        fp += fp_temp
        fn += fn_temp

    metrics_filtered = calculate_metrics(tp, tn, fp, fn)
    #
    # with mlflow.start_run(experiment_id=idExperiment,
    #                       run_name=input_path.split("/")[1] + "_" + filter_mask_path.split("/")[1]):
    print("MODEL_SEGMENTATION", input_path.split("/")[1])
    print("MODEL_CLASSIFICATION", filter_mask_path.split("/")[1])
    print(metrics)
    print(metrics_filtered)
    # mlflow.log_artifact("/content/image_final.png")
    for key in metrics:
        print("TEST_" + key, metrics[key])
    for key in metrics_filtered:
        print("TEST_" + key + "_FILTERED", metrics_filtered[key])

OUTPUT_DIR_PRE = "/Users/minajafari/PycharmProjects/Baseline_segmentation/output/PRE/"
OUTPUT_DIR_CRI = "/Users/minajafari/PycharmProjects/Baseline_segmentation/output/PRE/ViT/"
OUTPUT_DIR_CPF = "./output/CPF/"

data_dir = "/Users/minajafari/PycharmProjects/Baseline_segmentation/"
groundtruth_path = os.path.join(data_dir, "mask_tif")


list_results_PRE = os.listdir(OUTPUT_DIR_PRE)
list_results_CRI = os.listdir(OUTPUT_DIR_CRI)

print(list_results_PRE)
print(list_results_CRI)

if not os.path.isdir(OUTPUT_DIR_CPF):
    os.makedirs (OUTPUT_DIR_CPF)

# for result_PRE in list_results_PRE:
#     for result_CRI in list_results_CRI:
output_join_path = os.path.join(OUTPUT_DIR_CPF)
input_path = os.path.join(OUTPUT_DIR_PRE, 'UnetPlusPlus-timm-efficientnet-b7-imagenet-all_patches')

filter_mask_path = os.path.join(OUTPUT_DIR_CRI, 'mask')
log_metrics(input_path, filter_mask_path, groundtruth_path, output_join_path)




