"""
Per-Class Metrics & Real vs Counterfeit Pair Analysis (Sections 16 & 18).
Prints structured AP tables and computes paired discrimination indices.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from datasets import CLASS_NAMES, PAIRED_CATEGORIES


def format_per_class_table(metrics_dict: Dict[str, Any], class_names: Optional[List[str]] = None) -> str:
    """
    Formats per-class evaluation results into a clean tabular string.
    """
    classes = class_names or CLASS_NAMES
    per_class_aps = metrics_dict["per_class_aps"]
    total_gts = metrics_dict.get("total_gt_per_class", {})

    rows = []
    for c_id, name in enumerate(classes):
        ap_dict = per_class_aps.get(c_id, {})
        ap50 = ap_dict.get(0.50, 0.0)
        ap75 = ap_dict.get(0.75, 0.0)
        
        # Mean across 0.50:0.95
        all_aps = [v for k, v in ap_dict.items() if not np.isnan(v)]
        ap50_95 = np.mean(all_aps) if all_aps else 0.0
        gt_count = total_gts.get(c_id, 0)

        rows.append({
            "Class ID": c_id,
            "Class Name": name,
            "GT Count": gt_count,
            "AP@0.50": f"{ap50:.4f}" if not np.isnan(ap50) else "N/A",
            "AP@0.75": f"{ap75:.4f}" if not np.isnan(ap75) else "N/A",
            "AP@[0.50:0.95]": f"{ap50_95:.4f}" if not np.isnan(ap50_95) else "N/A"
        })

    df = pd.DataFrame(rows)
    return df.to_string(index=False)


def analyze_real_vs_counterfeit(metrics_dict: Dict[str, Any], class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Analyzes paired performance between Authentic and Counterfeit categories.
    Computes delta AP and material discrimination capability.
    """
    classes = class_names or CLASS_NAMES
    per_class_aps = metrics_dict["per_class_aps"]

    pair_results = []
    for real_id, fake_id in PAIRED_CATEGORIES:
        real_name = classes[real_id]
        fake_name = classes[fake_id]

        real_ap50 = per_class_aps.get(real_id, {}).get(0.50, 0.0)
        fake_ap50 = per_class_aps.get(fake_id, {}).get(0.50, 0.0)

        # Average pair AP
        avg_pair_ap = (real_ap50 + fake_ap50) / 2.0
        delta_ap = abs(real_ap50 - fake_ap50)

        pair_results.append({
            "pair": f"{real_name} / {fake_name}",
            "real_class": real_name,
            "fake_class": fake_name,
            "real_ap50": real_ap50,
            "fake_ap50": fake_ap50,
            "avg_ap50": avg_pair_ap,
            "discrepancy": delta_ap
        })

    return {
        "pairs": pair_results,
        "mean_pair_ap50": float(np.mean([p["avg_ap50"] for p in pair_results]))
    }
