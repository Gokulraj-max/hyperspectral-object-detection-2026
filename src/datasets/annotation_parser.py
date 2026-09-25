"""
Annotation parser supporting COCO JSON, YOLO TXT, VOC XML, and CSV formats.
Converts bounding boxes into standard [x1, y1, x2, y2] absolute coordinates.
"""

import os
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


def convert_bbox(
    box: List[float],
    from_format: str,
    to_format: str,
    image_size: Optional[Tuple[int, int]] = None
) -> List[float]:
    """
    Converts bounding box between formats:
    - 'xyxy': [x1, y1, x2, y2]
    - 'xywh': [x1, y1, w, h] (COCO format)
    - 'cxcywh': [cx, cy, w, h] (YOLO format)
    Can also handle normalization when image_size=(H, W) is provided.
    """
    box = [float(v) for v in box]
    
    # First convert to xyxy
    if from_format == "xyxy":
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    elif from_format == "xywh":
        x1, y1, w, h = box[0], box[1], box[2], box[3]
        x2 = x1 + w
        y2 = y1 + h
    elif from_format == "cxcywh":
        cx, cy, w, h = box[0], box[1], box[2], box[3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
    elif from_format == "yolo_norm":
        if image_size is None:
            raise ValueError("image_size (H, W) required for yolo_norm")
        h_img, w_img = image_size
        cx, cy, bw, bh = box[0] * w_img, box[1] * h_img, box[2] * w_img, box[3] * h_img
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
    else:
        raise ValueError(f"Unknown input format: {from_format}")

    # Now convert from xyxy to target format
    if to_format == "xyxy":
        return [x1, y1, x2, y2]
    elif to_format == "xywh":
        return [x1, y1, x2 - x1, y2 - y1]
    elif to_format == "cxcywh":
        return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]
    elif to_format == "yolo_norm":
        if image_size is None:
            raise ValueError("image_size (H, W) required for yolo_norm conversion")
        h_img, w_img = image_size
        cx = (x1 + x2) / (2.0 * w_img)
        cy = (y1 + y2) / (2.0 * h_img)
        bw = (x2 - x1) / w_img
        bh = (y2 - y1) / h_img
        return [cx, cy, bw, bh]
    else:
        raise ValueError(f"Unknown output format: {to_format}")


def parse_coco_json(json_path: str) -> Dict[str, Dict[str, Any]]:
    """Parse COCO format JSON annotations."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images = {img['id']: img for img in data.get('images', [])}
    id_to_filename = {img['id']: os.path.splitext(os.path.basename(img['file_name']))[0] for img in images.values()}
    
    parsed = {}
    for img_id, img_info in images.items():
        base_id = id_to_filename[img_id]
        parsed[base_id] = {
            "image_id": base_id,
            "width": img_info.get("width", None),
            "height": img_info.get("height", None),
            "boxes": [],
            "labels": []
        }

    for ann in data.get('annotations', []):
        img_id = ann['image_id']
        base_id = id_to_filename.get(img_id, str(img_id))
        if base_id not in parsed:
            continue
        bbox_xywh = ann['bbox']
        bbox_xyxy = convert_bbox(bbox_xywh, from_format="xywh", to_format="xyxy")
        category_id = int(ann['category_id'])
        parsed[base_id]["boxes"].append(bbox_xyxy)
        parsed[base_id]["labels"].append(category_id)

    return parsed


def parse_yolo_txt_dir(labels_dir: str, image_sizes: Optional[Dict[str, Tuple[int, int]]] = None) -> Dict[str, Dict[str, Any]]:
    """Parse YOLO txt files directory where each txt contains '<class_id> <cx> <cy> <w> <h>'."""
    parsed = {}
    for filename in os.listdir(labels_dir):
        if not filename.endswith(".txt"):
            continue
        image_id = os.path.splitext(filename)[0]
        filepath = os.path.join(labels_dir, filename)
        boxes, labels = [], []
        
        img_size = image_sizes.get(image_id, (512, 512)) if image_sizes else (512, 512)
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(float(parts[0]))
                    coords = [float(p) for p in parts[1:5]]
                    xyxy = convert_bbox(coords, from_format="yolo_norm", to_format="xyxy", image_size=img_size)
                    boxes.append(xyxy)
                    labels.append(cls_id)
        
        parsed[image_id] = {
            "image_id": image_id,
            "boxes": boxes,
            "labels": labels,
            "height": img_size[0],
            "width": img_size[1]
        }
    return parsed


def parse_csv_annotations(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """Parse CSV annotations with columns [image_id, class_id, x1, y1, x2, y2]."""
    df = pd.read_csv(csv_path)
    parsed = {}
    
    # Normalize column names
    col_map = {col.lower(): col for col in df.columns}
    img_col = col_map.get("image_id", col_map.get("image", None))
    cls_col = col_map.get("class_id", col_map.get("class", col_map.get("label", None)))
    x1_col = col_map.get("x1", col_map.get("xmin", None))
    y1_col = col_map.get("y1", col_map.get("ymin", None))
    x2_col = col_map.get("x2", col_map.get("xmax", None))
    y2_col = col_map.get("y2", col_map.get("ymax", None))

    if not all([img_col, cls_col, x1_col, y1_col, x2_col, y2_col]):
        raise ValueError(f"CSV missing essential detection columns. Found: {list(df.columns)}")

    for _, row in df.iterrows():
        img_id = str(row[img_col])
        if img_id not in parsed:
            parsed[img_id] = {
                "image_id": img_id,
                "boxes": [],
                "labels": []
            }
        box = [float(row[x1_col]), float(row[y1_col]), float(row[x2_col]), float(row[y2_col])]
        parsed[img_id]["boxes"].append(box)
        parsed[img_id]["labels"].append(int(row[cls_col]))

    return parsed


def parse_annotations(path: str, image_sizes: Optional[Dict[str, Tuple[int, int]]] = None) -> Dict[str, Dict[str, Any]]:
    """Unified entry point to parse annotations from JSON, CSV, or directory of YOLO txt files."""
    if os.path.isfile(path):
        if path.endswith(".json"):
            return parse_coco_json(path)
        elif path.endswith(".csv"):
            return parse_csv_annotations(path)
    elif os.path.isdir(path):
        # Look for annotations.json or annotations.csv inside directory
        json_candidate = os.path.join(path, "annotations.json")
        csv_candidate = os.path.join(path, "annotations.csv")
        if os.path.isfile(json_candidate):
            return parse_coco_json(json_candidate)
        if os.path.isfile(csv_candidate):
            return parse_csv_annotations(csv_candidate)
        # Otherwise treat as YOLO directory
        return parse_yolo_txt_dir(path, image_sizes=image_sizes)
    raise FileNotFoundError(f"Cannot identify annotation source at {path}")
