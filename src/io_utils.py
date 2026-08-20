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
                 labels: list, paths: list) -> Path:
    """เซฟ scores_{split_name}.npz ด้วย schema เดียวกับ repo หลัก
    (Anomaly-Detection-THESIS) เพื่อให้ script ที่โหลด .npz จากทั้งสอง repo
    มาเทียบกันใช้ key เดียวกันได้โดยไม่มี silent mismatch

    schema ที่บันทึก:
      scores  (float32, [N])    : anomaly score ต่อภาพ — ค่าสูง = ผิดปกติ
                                  ใช้กับ threshold ตอน inference จริง
      y_true  (int, [N])        : ground-truth label: 0=normal, 1=anomaly —
                                  KEY นี้ใช้คำนวณ metric ทุกตัว (AUROC,
                                  escape_rate ฯลฯ) ผ่าน compute_metrics()
                                  ห้ามสับสนกับ 'labels' ที่เป็น string
      labels  (string array, [N]): ชื่อ class แบบ string เช่น "good"/"defect" —
                                  ใช้สำหรับ display/gallery และ predictions
                                  CSV ไม่ใช้คำนวณ metric ใดเลย
      paths   (string array, [N]): path เต็มของภาพต้นฉบับ —
                                  ใช้ trace กลับตอน error analysis และ
                                  join กับ predictions CSV

    Save scores_{split_name}.npz with the same schema as the main repo
    (Anomaly-Detection-THESIS) so any script loading .npz from both repos
    can use the same key names without a silent mismatch.

    Schema saved:
      scores  (float32, [N])     : per-image anomaly score — higher = more
                                   anomalous; used against the threshold at
                                   real inference time
      y_true  (int, [N])         : ground-truth label: 0=normal, 1=anomaly —
                                   THIS key is used for all metric computation
                                   (AUROC, escape_rate, etc.) via
                                   compute_metrics(). Never confuse with
                                   'labels', which is a string.
      labels  (string array, [N]): class name string, e.g. "good"/"defect" —
                                   for display/gallery and predictions CSV
                                   only; never used for metric computation
      paths   (string array, [N]): full source image path —
                                   used for error analysis trace-back and
                                   joining with predictions CSV
    """
    out_path = Path(cfg.OUTPUT_PATH) / f"scores_{split_name}.npz"
    np.savez(
        out_path,
        scores = scores.astype(np.float32),
        y_true = y_true.astype(np.int64),
        labels = np.array(labels),
        paths  = np.array(paths),
    )
    return out_path