"""
One-shot entry point — แก้ OVERRIDES ด้านล่างแล้วรัน `python RUN.py`
เหมือน RUN.py ของ repo หลัก (Anomaly-Detection-THESIS)
"""
from config.config import Config
from scripts.run_patchcore import run

OVERRIDES = dict(
    # ── แก้ตรงนี้ก่อนรันจริง ─────────────────────────────────────────
    DATA_ROOT="dataset root path (contains good/ and defect/ subfolders)",
    GOOD_DIRNAME  = "good",
    DEFECT_DIRNAME = "defect",
    # แนะนำ: ชี้ไปที่ splits/split_assignment.csv เดียวกับ repo หลัก
    # (Anomaly-Detection-THESIS) เพื่อให้ train/val/test membership ตรงกัน
    SPLIT_CACHE_PATH="splits/split_assignment.csv",
    EXPERIMENT="PatchCore_group1_wide_resnet50_2",

    # ── ปรับได้ตามต้องการ ────────────────────────────────────────────
    BACKBONE="wide_resnet50_2",
    CORESET_RATIO=0.01,
    THRESHOLD_PERCENTILE=95.0,
)

if __name__ == "__main__":
    cfg = Config(**OVERRIDES)
    run(cfg)
