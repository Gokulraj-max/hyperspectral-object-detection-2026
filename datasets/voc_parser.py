"""
Pascal VOC XML Annotation Parser for Hyperspectral Object Detection Challenge 2026.
Parses VOC XML format:
<annotation>
    <filename>...</filename>
    <size>
        <width>...</width>
        <height>...</height>
        <depth>16</depth>
    </size>
    <object>
        <name>class_name</name>
        <bndbox>
            <xmin>...</xmin>
            <ymin>...</ymin>
            <xmax>...</xmax>
            <ymax>...</ymax>
        </bndbox>
    </object>
</annotation>
"""

import os
import glob
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Official 18 categories
CLASS_NAMES = [
    "real_leather", "fake_leather",
    "real_silk", "synthetic_silk",
    "real_gemstone", "glass_gemstone",
    "real_banknote", "counterfeit_banknote",
    "real_wood", "laminated_wood",
    "real_wool", "synthetic_wool",
    "real_jade", "counterfeit_jade",
    "authentic_chip", "counterfeit_chip",
    "natural_honey", "adulterated_honey"
]

CLASS_TO_ID = {name.lower().strip(): i for i, name in enumerate(CLASS_NAMES)}


def parse_voc_annotation(xml_path: str, class_to_id: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """
    Parses a single VOC XML file.
    Returns:
    {
        'image_id': str,
        'boxes': [[x1, y1, x2, y2], ...],
        'labels': [class_id, ...],
        'width': int,
        'height': int,
        'depth': int
    }
    """
    c_map = class_to_id or CLASS_TO_ID
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_id = os.path.splitext(os.path.basename(xml_path))[0]
    filename_elem = root.find("filename")
    if filename_elem is not None and filename_elem.text:
        image_id = os.path.splitext(os.path.basename(filename_elem.text.strip()))[0]

    size_elem = root.find("size")
    width, height, depth = None, None, 16
    if size_elem is not None:
        w_text = size_elem.findtext("width")
        h_text = size_elem.findtext("height")
        d_text = size_elem.findtext("depth")
        if w_text: width = int(float(w_text))
        if h_text: height = int(float(h_text))
        if d_text: depth = int(float(d_text))

    boxes = []
    labels = []

    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None or not name_elem.text:
            continue
        class_name = name_elem.text.strip().lower()

        # Handle variations or integer indices
        if class_name in c_map:
            cls_id = c_map[class_name]
        elif class_name.isdigit() and int(class_name) in range(len(CLASS_NAMES)):
            cls_id = int(class_name)
        elif class_name.startswith("class_") and class_name.split("_")[-1].isdigit():
            cls_id = int(class_name.split("_")[-1])
        else:
            # Fuzzy match
            found = False
            for cand, idx in c_map.items():
                if cand in class_name or class_name in cand:
                    cls_id = idx
                    found = True
                    break
            if not found:
                continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = float(bndbox.findtext("xmin"))
            ymin = float(bndbox.findtext("ymin"))
            xmax = float(bndbox.findtext("xmax"))
            ymax = float(bndbox.findtext("ymax"))

            # Clip coordinate validity
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(cls_id)
        except (TypeError, ValueError):
            continue

    return {
        "image_id": image_id,
        "boxes": boxes,
        "labels": labels,
        "width": width,
        "height": height,
        "depth": depth
    }


def parse_voc_directory(annotations_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Parses an entire directory of VOC XML files.
    Returns mapping: image_id -> annotation dict
    """
    annotations = {}
    xml_files = glob.glob(os.path.join(annotations_dir, "*.xml"))
    for xml_path in xml_files:
        try:
            data = parse_voc_annotation(xml_path)
            stem = os.path.splitext(os.path.basename(xml_path))[0]
            annotations[stem] = data
            if data["image_id"] != stem:
                annotations[data["image_id"]] = data
        except Exception as e:
            print(f"[WARN] Failed parsing {xml_path}: {e}")
    return annotations
