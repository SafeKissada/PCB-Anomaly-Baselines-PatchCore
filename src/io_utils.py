import json
from pathlib import Path

import numpy as np


def save_final_results(cfg, split_name: str, metrics: dict, threshold: float,
                        extra: dict = None) -> Path:
    """เซฟ final_results.json ในรูปแบบเดียวกับ repo หลัก (config snapshot +
    metrics) เพื่อให้เขียนสคริปต์เทียบผลข้าม repo ได้ง่าย
    """
    out = {
        "experiment": cfg.EXPERIMENT,
        "backbone": cfg.BACKBONE,
        "split": split_name,
        "threshold": threshold,
        "threshold_percentile": cfg.THRESHOLD_PERCENTILE,
        "coreset_ratio": cfg.CORESET_RATIO,
        "metrics": {k: v for k, v in metrics.items()
                    if k not in ("cm", "fpr", "tpr", "gt", "pred", "scores")},
        "confusion_matrix": metrics["cm"].tolist(),
    }
    if extra:
        out.update(extra)

    out_path = Path(cfg.OUTPUT_PATH) / f"final_results_{split_name}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def save_scores(cfg, split_name: str, scores: np.ndarray, y_true: np.ndarray,
                 labels: list, paths: list,
                 heatmaps: np.ndarray,
                 orig_imgs: np.ndarray,
                 preproc_imgs: np.ndarray) -> Path:
    """เซฟ scores_{split_name}.npz ด้วย schema เดียวกับ repo หลัก
    (Anomaly-Detection-THESIS) เพื่อให้ visualize.py และ script เทียบผล
    ข้าม repo ใช้ key เดียวกันได้โดยไม่มี silent mismatch

    schema ที่บันทึก (ตรงกับ io_utils.save_scores ของ repo หลักทุก key):
      scores       (float32, [N])       : anomaly score ต่อภาพ
      y_true       (int64,   [N])       : ground-truth label 0/1
      labels       (string,  [N])       : ชื่อ class "good"/"defect"
      paths        (string,  [N])       : path ต้นฉบับ
      heatmaps     (float32, [N, H, W]) : pixel-level anomaly heatmap
                                          (kNN distance map หลัง upsample+smooth)
      orig_imgs    (float32, [N, H, W, 3]): ภาพ RGB ต้นฉบับ ก่อน normalize
      preproc_imgs (float32, [N, H, W, 3]): ภาพหลัง preprocessing จริง

    Save scores_{split_name}.npz with the same schema as the main repo
    (Anomaly-Detection-THESIS) so visualize.py and cross-repo comparison
    scripts can use the same key names without any silent mismatch.

    Schema (matches the main repo's io_utils.save_scores key-for-key):
      scores       (float32, [N])        : per-image anomaly score
      y_true       (int64,   [N])        : ground-truth label 0/1
      labels       (string,  [N])        : class name "good"/"defect"
      paths        (string,  [N])        : source file path
      heatmaps     (float32, [N, H, W])  : pixel-level anomaly heatmap
                                           (kNN distance map after upsample+smooth)
      orig_imgs    (float32, [N, H, W, 3]): original RGB image before normalization
      preproc_imgs (float32, [N, H, W, 3]): image after real preprocessing
    """
    out_path = Path(cfg.OUTPUT_PATH) / f"scores_{split_name}.npz"
    np.savez_compressed(
        out_path,
        scores       = scores.astype(np.float32),
        y_true       = y_true.astype(np.int64),
        labels       = np.array(labels),
        paths        = np.array(paths),
        heatmaps     = heatmaps.astype(np.float32),
        orig_imgs    = orig_imgs.astype(np.float32),
        preproc_imgs = preproc_imgs.astype(np.float32),
    )
    return out_path