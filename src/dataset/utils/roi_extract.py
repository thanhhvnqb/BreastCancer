import os
import sys

import cv2
import numpy as np


def extract_roi_otsu(img, gkernel=(5, 5)):
    """WARNING: this function modify input image inplace."""
    ori_h, ori_w = img.shape[:2]
    # clip percentile: implant, white lines
    upper = np.percentile(img, 95)
    img[img > upper] = np.min(img)
    # Gaussian filtering to reduce noise (optional)
    if gkernel is not None:
        img = cv2.GaussianBlur(img, gkernel, 0)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # dilation to improve contours connectivity
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (-1, -1))
    img_bin = cv2.dilate(img_bin, element)
    cnts, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None, None, None
    areas = np.array([cv2.contourArea(cnt) for cnt in cnts])
    select_idx = np.argmax(areas)
    cnt = cnts[select_idx]
    area_pct = areas[select_idx] / (img.shape[0] * img.shape[1])
    x0, y0, w, h = cv2.boundingRect(cnt)
    # min-max for safety only
    # x0, y0, x1, y1
    x1 = min(max(int(x0 + w), 0), ori_w)
    y1 = min(max(int(y0 + h), 0), ori_h)
    x0 = min(max(int(x0), 0), ori_w)
    y0 = min(max(int(y0), 0), ori_h)
    return [x0, y0, x1, y1], area_pct, None


def extract_roi_fit_breast(img):
    # Some images have narrow exterior "frames" that complicate selection of the main data. Cutting off the frame
    img = img[5:-5, 5:-5]

    # regions of non-empty pixels
    output = cv2.connectedComponentsWithStats(
        (img > 0.05).astype(np.uint8)[:, :], 8, cv2.CV_32S
    )

    # stats.shape == (N, 5), where N is the number of regions, 5 dimensions correspond to:
    # left, top, width, height, area_size
    stats = output[2]

    # finding max area which always corresponds to the breast data.
    idx = stats[1:, 4].argmax() + 1
    x1, y1, w, h = stats[idx][:4]
    x2 = x1 + w
    y2 = y1 + h

    # cutting out the breast data
    area_pct = img[y1:y2, x1:x2]

    return [x1, y1, x2, y2], area_pct, None


class RoiExtractor:

    def __init__(
        self,
        area_pct_thres=0.04,
    ):
        self.area_pct_thres = area_pct_thres

    def detect_single(self, img):
        xyxy, _, _ = extract_roi_fit_breast(img)
        if xyxy is not None:
            x0, y0, x1, y1 = xyxy
            return [x0, y0, x1, y1], None, None
        print("ROI detection using fit Breast fail.")
        sys.exit()
        return None, None, None

    def detect_single_otsu(self, img):
        xyxy, area_pct, _ = extract_roi_otsu(img)
        # if both fail, use full frame
        if xyxy is not None:
            if area_pct >= self.area_pct_thres:
                print("ROI detection: using Otsu.")
                x0, y0, x1, y1 = xyxy
                return [x0, y0, x1, y1], area_pct, None
        print("ROI detection using Otsu fail.")
        sys.exit()
        return None, area_pct, None
