"""
รัน PatchCore แบบ end-to-end บน dataset เดียวกับ repo หลัก
(Anomaly-Detection-THESIS) แล้ว log metric ให้เทียบกับ EXPERIMENT 0
(ConvNeXt+AE) ได้ตรงๆ

ฟังก์ชัน run() ออกแบบให้ถูกเรียกได้จากหลายที่:
  - RUN.py          → รันรอบเดียวด้วย config จาก OVERRIDES
  - RUN_multi_seed.py → รันซ้ำหลาย seed โดย reuse OVERRIDES เดิม

split assignment (train/val/test membership) ถูกจัดการโดย
build_datasets_and_loaders() ผ่าน cfg.SPLIT_CACHE_PATH เพียงจุดเดียว:
  - ถ้าไฟล์ cache ยังไม่มี → คำนวณ split ใหม่แล้วเซฟไว้
  - ถ้ามีอยู่แล้ว → โหลดจาก cache โดยตรง ไม่คำนวณซ้ำ
ทำให้ multi-seed ที่แชร์ SPLIT_CACHE_PATH เดียวกันได้ train/val/test
membership เดิมทุก seed (variance มาจาก coreset randomness เท่านั้น)
ส่วน multi-seed ที่แยก SPLIT_CACHE_PATH ต่อ seed จะได้ split ต่างกัน
(variance รวมทั้ง split และ coreset randomness)

Usage:
    python RUN.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import Config, set_seed
from src.data.dataset import build_datasets_and_loaders
from src.evaluate import compute_metrics, select_percentile_threshold
from src.io_utils import save_final_results, save_scores
from src.models.patchcore import PatchCore

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("run_patchcore")


def run(cfg: Config):
    """รัน PatchCore 1 รอบเต็ม: split → fit → score → threshold → save

    split assignment ถูกจัดการโดย build_datasets_and_loaders() ผ่าน
    cfg.SPLIT_CACHE_PATH — ถ้า cache ยังไม่มีจะคำนวณใหม่แล้วเซฟ ถ้ามีแล้ว
    จะโหลดโดยตรง การ cache นี้ทำให้ multi-seed ที่ชี้ SPLIT_CACHE_PATH
    เดียวกันได้ split เหมือนกันทุก seed โดยอัตโนมัติ

    split assignment is handled entirely by build_datasets_and_loaders()
    via cfg.SPLIT_CACHE_PATH — computed and saved if the cache doesn't
    exist yet, loaded directly if it does. This caching means multi-seed
    runs pointing at the same SPLIT_CACHE_PATH automatically get the
    same split every seed.
    """
    set_seed(cfg.SEED)

    logger.info(
        f"Loading data จาก {cfg.DATA_ROOT} "
        f"(split cache: {cfg.SPLIT_CACHE_PATH})"
    )
    data = build_datasets_and_loaders(cfg)
    logger.info(
        f"Train (normal เท่านั้น): {len(data['df_train'])} ภาพ | "
        f"Val: {len(data['df_val'])} | Test: {len(data['df_test'])}"
    )

    model = PatchCore(cfg)
    model.fit(data["normal_loader"])

    val_result  = model.score(data["val_loader"])
    test_result = model.score(data["test_loader"])

    # y_true (int 0/1) ใช้คำนวณ metric — ห้ามสลับกับ result.labels (string)
    # เพราะ select_percentile_threshold() และ compute_metrics() expect int 0/1
    # ไม่ใช่ string "good"/"defect" (ดู src/models/base.py สำหรับนิยาม field)
    #
    # y_true (int 0/1) is for metric computation — never swap with
    # result.labels (string): select_percentile_threshold() and
    # compute_metrics() expect int 0/1, not the string "good"/"defect"
    # (see src/models/base.py for the field definitions).
    threshold = select_percentile_threshold(
        val_result.image_scores, val_result.y_true, cfg)
    logger.info(
        f"Threshold (percentile={cfg.THRESHOLD_PERCENTILE}): {threshold:.6f}"
    )

    for split_name, result in [("val", val_result), ("test", test_result)]:
        metrics = compute_metrics(
            result.image_scores, result.y_true, threshold)
        logger.info(
            f"[{split_name}] AUC={metrics['auc']:.4f}  "
            f"AP={metrics['ap']:.4f}  Acc={metrics['acc']:.4f}  "
            f"Prec={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"EscapeRate={metrics['escape_rate']:.4f}  "
            f"AutoClearRate={metrics['auto_clear_rate']:.4f}"
        )
        save_final_results(cfg, split_name, metrics, threshold)
        save_scores(
            cfg, split_name,
            result.image_scores, result.y_true,
            result.labels, result.paths,
            result.pixel_maps,
            result.orig_imgs,
            result.preproc_imgs,
        )

    logger.info(f"ผลลัพธ์ทั้งหมดถูกเซฟไว้ที่ {cfg.OUTPUT_PATH}")
    return val_result, test_result


if __name__ == "__main__":
    cfg = Config()
    run(cfg)