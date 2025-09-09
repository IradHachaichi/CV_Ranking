import os
import cv2
import torch
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict

from craft_utils import getDetBoxes, adjustResultCoordinates
import imgproc
from craft import CRAFT
from file_utils import saveResult


def copy_state_dict(state_dict):
    start_idx = 1 if list(state_dict.keys())[0].startswith("module") else 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def load_craft_model(model_path, use_cuda=True):
    net = CRAFT()
    state = torch.load(model_path, map_location='cuda' if use_cuda else 'cpu')
    net.load_state_dict(copy_state_dict(state))
    if use_cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = False
    net.eval()
    return net


def detect_text_boxes(image, net, text_threshold=0.7, link_threshold=0.2, low_text=0.4,
                      use_cuda=True, canvas_size=1280, mag_ratio=1.5, poly=False, refine_net=None):
    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    if use_cuda:
        x = x.cuda()

    with torch.no_grad():
        y, feature = net(x)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)

    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    return boxes, polys, score_text


def group_boxes_nearby(boxes, max_horizontal_gap=20, max_vertical_diff=10):
    rects = []
    for box in boxes:
        x_coords = box[:, 0]
        y_coords = box[:, 1]
        rects.append([np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords)])
    rects = np.array(rects)
    indices = np.lexsort((rects[:, 0], rects[:, 1]))
    rects = rects[indices]

    groups = []
    current_group = [rects[0]]

    for rect in rects[1:]:
        prev = current_group[-1]
        vertical_diff = abs((rect[1] + rect[3]) / 2 - (prev[1] + prev[3]) / 2)
        horizontal_gap = rect[0] - prev[2]
        if vertical_diff <= max_vertical_diff and 0 <= horizontal_gap <= max_horizontal_gap:
            current_group.append(rect)
        else:
            groups.append(current_group)
            current_group = [rect]
    groups.append(current_group)
    return groups


def merge_groups(groups):
    merged = []
    for group in groups:
        group = np.array(group)
        merged.append([
            np.min(group[:, 0]),
            np.min(group[:, 1]),
            np.max(group[:, 2]),
            np.max(group[:, 3])
        ])
    return merged


def detect_text_lines(image_path, model_path, result_dir=None,
                      text_threshold=0.7, link_threshold=0.2, low_text=0.4,
                      canvas_size=1280, mag_ratio=1.5,
                      max_horizontal_gap=300, max_vertical_diff=80,
                      use_cuda=None):
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()

    net = load_craft_model(model_path, use_cuda)
    image = imgproc.loadImage(image_path)

    boxes, polys, score_text = detect_text_boxes(
        image, net,
        text_threshold=text_threshold,
        link_threshold=link_threshold,
        low_text=low_text,
        use_cuda=use_cuda,
        canvas_size=canvas_size,
        mag_ratio=mag_ratio
    )

    groups = group_boxes_nearby(boxes, max_horizontal_gap, max_vertical_diff)
    merged_boxes = merge_groups(groups)

    line_images = []
    for i, (x_min, y_min, x_max, y_max) in enumerate(merged_boxes):
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        crop = image[y_min:y_max, x_min:x_max]
        if crop.shape[0] > 5 and crop.shape[1] > 5:
            line_images.append(crop)
            if result_dir:
                os.makedirs(result_dir, exist_ok=True)
                cv2.imwrite(os.path.join(result_dir, f"line_{i+1}.png"), crop)

    return merged_boxes, line_images
