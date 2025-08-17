import torch
import cv2
import pickle
import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import random
import math
import time
import glob
import albumentations as albu
from model import YOLOV1

# defining model directory 
model_dir = "Model/yolo_v1_best_on_voc.pt"

# model parameters
model_config = {
    'im_channels': 3,
    'backbone_channels': 512,
    'conv_spatial_size': 7,
    'yolo_conv_channels': 1024,
    'leaky_relu_slope': 0.1,
    'fc_dim': 1024,
    'fc_dropout': 0.5,
    'S': 7,
    'B': 2,
    'use_sigmoid': True,
    'use_conv': True 
}

# define YOLO-V1
model = YOLOV1(448, 20, model_config)

# device
# device= 'cpu'
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# load pre-trained weights
model.load_state_dict(torch.load(model_dir, map_location=device, weights_only=True))

def draw_boxes_on_image(image, boxes, labels, scores, obj_to_idx, score_threshold=0.3):
    """
    Draw bounding boxes on the image with class labels and confidence scores.
    Args:
        image (ndarray): The input image to draw the bounding boxes on.
        boxes (Tensor): The predicted bounding boxes (xmin, ymin, xmax, ymax).
        labels (Tensor): The predicted class labels (indices from obj_to_idx).
        scores (Tensor): The confidence scores for each prediction.
        obj_to_idx (dict): Mapping of class names to indices.
        score_threshold (float): Minimum score threshold to display the boxes.
    Returns:
        image (ndarray): The image with the drawn bounding boxes.
    """
    # Convert the image to the format compatible with OpenCV (if necessary)
    if isinstance(image, Image.Image):
        image = np.array(image)
        
    image = image.astype(np.uint8)

    for i in range(len(boxes)):
        # Only draw boxes with a score above the threshold
        if scores[i] > score_threshold:
            xmin, ymin, xmax, ymax = boxes[i]

            # Get the class name using the label index from obj_to_idx
            label_idx = labels[i]  # Convert tensor to Python number
            class_name = [key for key, value in obj_to_idx.items() if value == label_idx][0]

            # Draw the bounding box and label on the image
            color = (0, 255, 0)  # Green color for the box
            thickness = 2
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)

            # Label text
            label_text = f'{class_name}: {scores[i]:.2f}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            cv2.putText(image, label_text, (int(xmin), int(ymin)-10), font, font_scale, color, font_thickness)

    return image

def convert_yolo_pred_to_box(yolo_pred, S, B, C, use_sigmoid=False):
    r"""
    Method converts yolo predictions to
    x1y1x2y2 format
    """
    out = yolo_pred.reshape((S, S, 5 * B + C))
    if use_sigmoid:
        out[..., :5 * B] = torch.nn.functional.sigmoid(out[..., :5 * B])
    out = torch.clamp(out, min=0., max=1.)
    class_score, class_idx = torch.max(out[..., 5 * B:], dim=-1)

    # Create a grid using these shifts
    # Will use these for converting x_center_offset/y_center_offset
    # values to x1/y1/x2/y2(normalized 0-1)
    # S cells = 1 => each cell adds 1/S pixels of shift
    shifts_x = torch.arange(0, S, dtype=torch.int32, device=out.device) * 1 / float(S)
    shifts_y = torch.arange(0, S, dtype=torch.int32, device=out.device) * 1 / float(S)
    shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

    boxes = []
    confidences = []
    labels = []
    for box_idx in range(B):
        # xc_offset yc_offset w h -> x1 y1 x2 y2
        boxes_x1 = ((out[..., box_idx * 5] * 1 / float(S) + shifts_x) -
                    0.5 * torch.square(out[..., 2 + box_idx * 5])).reshape(-1, 1)
        boxes_y1 = ((out[..., 1 + box_idx * 5] * 1 / float(S) + shifts_y) -
                    0.5 * torch.square(out[..., 3 + box_idx * 5])).reshape(-1, 1)
        boxes_x2 = ((out[..., box_idx * 5] * 1 / float(S) + shifts_x) +
                    0.5 * torch.square(out[..., 2 + box_idx * 5])).reshape(-1, 1)
        boxes_y2 = ((out[..., box_idx * 5] * 1 / float(S) + shifts_y) +
                    0.5 * torch.square(out[..., 3 + box_idx * 5])).reshape(-1, 1)
        boxes.append(torch.cat([boxes_x1, boxes_y1, boxes_x2, boxes_y2], dim=-1))
        confidences.append((out[..., 4 + box_idx * 5] * class_score).reshape(-1))
        labels.append(class_idx.reshape(-1))
    boxes = torch.cat(boxes, dim=0)
    scores = torch.cat(confidences, dim=0)
    labels = torch.cat(labels, dim=0)
    return boxes, scores, labels

# object to index matching 
obj_to_idx = {
    'aeroplane': 0, 'bicycle': 1,
    'bird': 2, 'boat': 3,
    'bottle': 4, 'bus': 5,
    'car': 6, 'cat': 7,
    'chair': 8, 'cow': 9,
    'diningtable': 10, 'dog': 11,
    'horse': 12, 'motorbike': 13,
    'person': 14, 'pottedplant': 15,
    'sheep': 16, 'sofa': 17,
    'train': 18, 'tvmonitor': 19
}

# function to get detection 
def get_detection(model, org_img, input_img, conf_threshold=0.2, nms_threshold=0.5, obj_to_idx=obj_to_idx, score_threshold=0.5):
    """
    This function takes an input image and provide predicted object with bounding boxes
    Args:
        model
        org_img
        input_img
        use_sigmoid
        conf_threshold
        nms_threshold
        obj_to_idx
        score_threshold
    """
    model.to(device)
    model.eval()
    # Start the timer
    start_time = time.time()
    with torch.no_grad():
        pred = model(input_img)

    boxes, scores, labels =  convert_yolo_pred_to_box(pred, S=7, B=2, C=20, use_sigmoid=True) 
    
    # Confidence Score Thresholding 
    keep = torch.where(scores > conf_threshold)[0]
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    # NMS
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for class_id in torch.unique(labels):
        curr_indices = torch.where(labels == class_id)[0]
        curr_keep_indices = torch.ops.torchvision.nms(boxes[curr_indices],
                                                      scores[curr_indices],
                                                      nms_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep = torch.where(keep_mask)[0]
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Scale prediction boxes x1y1x2y2 from 0-1 to 0-w and 0-h
    org_h, org_w, org_c = org_img.shape
    box_rescaled = boxes.clone()
    box_rescaled[..., 0::2] = (org_w * box_rescaled[..., 0::2])
    box_rescaled[..., 1::2] = (org_h * box_rescaled[..., 1::2])

    # Calculate the time taken for the prediction
    inference_time_1 = time.time() - start_time
    
    # draw the bounding box
    result = draw_boxes_on_image(org_img, box_rescaled, labels, scores, obj_to_idx, score_threshold=score_threshold)
    # Calculate the time taken for the prediction
    inference_text = f"Inference time: {inference_time_1:.4f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Adding the inference time text to the image in red color and positioned on the left
    cv2.putText(result, inference_text, (10, 30), font, 0.7, (0, 0, 255), 2)  # Red color (0, 0, 255)
    # return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result

# function to convert input image into tensor 
def prepare_input(input_img, t_size):
    
    transforms = albu.Compose([
        albu.Resize(t_size, t_size)
    ])
    
    img = transforms(image=input_img)
    im_tensor = torch.from_numpy(img['image'] / 255.).permute((2,0,1)).float()
    im_tensor_cnl_0 = (torch.unsqueeze(im_tensor[0], 0) - 0.485) / 0.229
    im_tensor_cnl_1 = (torch.unsqueeze(im_tensor[1], 0) - 0.456) / 0.224
    im_tensor_cnl_2 = (torch.unsqueeze(im_tensor[2], 0) - 0.406) / 0.225
    im_tensor = torch.cat((im_tensor_cnl_0,
                            im_tensor_cnl_1,
                            im_tensor_cnl_2), 0)

    im_tensor = im_tensor.unsqueeze(0)
    return im_tensor.to(device)