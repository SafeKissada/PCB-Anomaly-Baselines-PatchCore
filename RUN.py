"""
One-shot entry point สำหรับ PatchCore — แก้ OVERRIDES ด้านล่างแล้วรัน:
    python RUN.py

เหมือน RUN.py ของ repo หลัก (Anomaly-Detection-THESIS) ทุกประการ:
  - OVERRIDES เป็นเพียงที่เดียวที่ต้องแก้ ไม่ต้องแตะไฟล์อื่น
  - script อื่นที่ต้องการ config เดิม (RUN_multi_seed.py) import
    OVERRIDES จากไฟล์นี้โดยตรง ไม่ copy ซ้ำ กัน 2 ไฟล์ไม่ sync กัน

เรื่อง split assignment (train/val/test membership):
  - ถูกจัดการโดย build_datasets_and_loaders() ผ่าน cfg.SPLIT_CACHE_PATH
    ภายใน run() เพียงจุดเดียว — ไม่มี "save_split" แยกต่างหาก
  - รอบแรกที่รัน: คำนวณ split ใหม่แล้วเซฟ cache ลง SPLIT_CACHE_PATH
  - รอบถัดไป (รวม multi-seed): โหลด cache โดยตรง ไม่คำนวณซ้ำ
  - multi-seed ที่ต้องการ split เดิมทุก seed: ชี้ SPLIT_CACHE_PATH
    เดียวกันทุก seed (แชร์ไฟล์ cache)
  - multi-seed ที่ต้องการ split ต่างกันต่อ seed: ให้ RUN_multi_seed.py
    แทน "SEED 42" ใน SPLIT_CACHE_PATH ด้วย seed จริง (แยก cache ต่อ seed)

About split assignment (train/val/test membership):
  - Entirely handled by build_datasets_and_loaders() via
    cfg.SPLIT_CACHE_PATH inside run() — there is no separate
    "save_split" step.
  - First run: computes a fresh split and saves it to SPLIT_CACHE_PATH.
  - Subsequent runs (including multi-seed): loads the cache directly,
    no recomputation.
  - Multi-seed wanting the SAME split every seed: point all seeds at
    the same SPLIT_CACHE_PATH (shared cache file).
  - Multi-seed wanting a DIFFERENT split per seed: let RUN_multi_seed.py
    replace "SEED 42" in SPLIT_CACHE_PATH with the real seed (separate
    cache per seed).
"""
from config.config import Config
from scripts.run_patchcore import run

OVERRIDES = dict(
    # ── Data & paths — แก้ก่อนรันจริง ──────────────────────────────
    DATA_ROOT="/content/drive/MyDrive/DATA/group 1",
    GOOD_DIRNAME="good",
    DEFECT_DIRNAME="defect",

    # แนะนำ: ชี้ไปที่ split_assignment.csv เดียวกับ repo หลัก
    # (Anomaly-Detection-THESIS) เพื่อให้ train/val/test membership
    # ตรงกันเป๊ะตอนเทียบ AE กับ PatchCore — ถ้าต้องการ split แยกต่อ
    # seed ให้ใส่ "SEED 42" ฝังไว้ใน path แล้ว RUN_multi_seed.py
    # จะแทนที่ให้อัตโนมัติเหมือน repo หลัก
    #
    # Recommended: point at the same split_assignment.csv as the main
    # repo (Anomaly-Detection-THESIS) so train/val/test membership
    # matches exactly when comparing AE vs PatchCore. To get a
    # different split per seed, embed "SEED 42" in the path and
    # RUN_multi_seed.py will substitute it automatically, same as
    # the main repo.
    SPLIT_CACHE_PATH="/content/drive/MyDrive/Result/PatchCore/SEED 42/splits/split_assignment.csv",
    SAVE_PATH="/content/drive/MyDrive/Result/PatchCore/SEED 42/log",
    OUTPUT_PATH="/content/drive/MyDrive/Result/PatchCore/SEED 42/image",
    SEED=42,

    # ── Model config — ปรับได้ตามต้องการ ────────────────────────────
    EXPERIMENT="PatchCore_group1_wide_resnet50_2",
    BACKBONE="wide_resnet50_2",
    CORESET_RATIO=0.01,
    THRESHOLD_PERCENTILE=95.0,
)

if __name__ == "__main__":
    cfg = Config(**OVERRIDES)
    run(cfg)