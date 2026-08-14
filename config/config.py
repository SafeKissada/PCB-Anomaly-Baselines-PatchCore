"""
Config สำหรับ baseline methods (PatchCore, และในอนาคต PaDiM/DRAEM/SimpleNet/RD4AD)

DATA_ROOT / GOOD_DIRNAME / DEFECT_DIRNAME / SPLIT_RATIOS / SPLIT_CACHE_PATH /
GROUP_ID_REGEX / SEED / VALID_EXT / IMAGE_SIZE / BATCH_SIZE / NUM_WORKERS /
PIN_MEMORY / THRESHOLD_PERCENTILE / HEATMAP_SIGMA / USE_GRAYSCALE* / USE_CLAHE*
ตั้งชื่อ field เหมือนกับ repo หลัก (Anomaly-Detection-THESIS) เป๊ะๆ โดยตั้งใจ —
เพื่อให้สามารถชี้ SPLIT_CACHE_PATH ไปที่ไฟล์ split เดียวกันได้ (แนะนำอย่างยิ่ง
ให้ทำแบบนี้ ดู README หัวข้อ "การเทียบผลกับ baseline เดิม") และให้ผลลัพธ์จาก
สอง repo เทียบกันได้ตรงๆ โดยไม่มี confound เรื่อง train/val/test membership
หรือนิยาม metric ต่างกัน
"""
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np
import torch


@dataclass
class Config:
    # ── Data (ต้องตรงกับ repo หลักถ้าจะเทียบผลกัน) ──────────────────
    DATA_ROOT: str = "dataset root path (contains good/ and defect/ subfolders)"
    GOOD_DIRNAME: str = "good"
    DEFECT_DIRNAME: str = "defect"
    SPLIT_RATIOS: Tuple[float, float, float] = (0.70, 0.15, 0.15)
    # ชี้ไปที่ splits/split_assignment.csv ของ repo หลักถ้ามีแล้ว เพื่อ reuse
    # split เดิมเป๊ะๆ (ห้ามลบ/สร้างใหม่ ไม่งั้น train/val/test membership จะ
    # ไม่ตรงกับ baseline ConvNeXt+AE อีกต่อไป)
    SPLIT_CACHE_PATH: str = "splits/split_assignment.csv"
    GROUP_ID_REGEX: Optional[str] = None
    VALID_EXT: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

    SAVE_PATH: str = "save/logs"
    OUTPUT_PATH: str = "save/results"

    # ── Reproducibility ──────────────────────────────────────────────
    SEED: int = 42
    DEVICE: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    EXPERIMENT: str = "PatchCore_Baseline"

    # ── Image / DataLoader (ต้องตรงกับ repo หลักถ้าจะเทียบผลกัน) ─────
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True
    USE_AUGMENTATION: bool = False  # PatchCore ไม่เทรน backbone จึงไม่จำเป็นต้องใช้
    AUG_COLOR_JITTER: float = 0.20

    # ── Color mode (เหมือน repo หลัก) ────────────────────────────────
    USE_GRAYSCALE: bool = False
    USE_GRAYSCALE_EQUALIZATION: bool = False
    USE_CLAHE: bool = False
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID_SIZE: tuple = (8, 8)

    # ── Evaluation (ต้องตรงกับ repo หลักถ้าจะเทียบผลกัน) ─────────────
    THRESHOLD_PERCENTILE: float = 95.0
    HEATMAP_SIGMA: float = 4.0

    # ── PatchCore-specific ────────────────────────────────────────────
    # Backbone สำหรับดึง feature ('wide_resnet50_2' ตาม paper ต้นฉบับ,
    # 'resnet18' เป็นตัวเบาสำหรับ debug/smoke-test เร็วๆ)
    BACKBONE: str = "wide_resnet50_2"
    # ต้องเป็น True เสมอตอนรันผลจริง (PatchCore ไม่เทรน backbone เลย พึ่ง
    # ImageNet feature ล้วนๆ) — False มีไว้สำหรับ smoke test/offline dev เท่านั้น
    PRETRAINED: bool = True
    # Layer ที่ดึง feature ออกมา (มาตรฐาน PatchCore = layer2 + layer3)
    FEATURE_LAYERS: Tuple[str, ...] = ("layer2", "layer3")
    # kernel ของ locally-aware patch feature (average pooling รอบ patch)
    PATCH_POOL_KERNEL: int = 3
    # สัดส่วน patch ที่เก็บไว้ใน memory bank หลัง greedy coreset subsampling
    # (0.01 = เก็บ 1% ตามค่า default ของ PatchCore paper บน MVTec)
    CORESET_RATIO: float = 0.01
    # มิติที่ projection ลงก่อนทำ greedy k-center (เร่งความเร็ว, ค่ามาตรฐาน paper)
    CORESET_PROJECTION_DIM: int = 128
    # จำนวน nearest neighbor ของ memory-bank point ที่ใช้ตอน re-weight
    # image-level score (b* ใน PatchCore paper, ค่ามาตรฐาน = 9 บน MVTec;
    # ปรับลงถ้า memory bank เล็กกว่านี้มาก)
    REWEIGHT_NUM_NEIGHBORS: int = 9
    # ขนาด batch ตอนคำนวณ pairwise distance กับ memory bank (กัน OOM)
    KNN_CHUNK_SIZE: int = 4096

    _DATA_ROOT_PLACEHOLDER = "dataset root path (contains good/ and defect/ subfolders)"

    @property
    def COLOR_MODE(self) -> str:
        if self.USE_GRAYSCALE_EQUALIZATION and self.USE_CLAHE:
            return "GRAYSCALE_EQUALIZATION_CLAHE"
        elif self.USE_GRAYSCALE_EQUALIZATION:
            return "GRAYSCALE_EQUALIZATION"
        elif self.USE_CLAHE:
            return "GRAYSCALE_CLAHE"
        elif self.USE_GRAYSCALE:
            return "GRAYSCALE"
        else:
            return "RGB"

    def __post_init__(self):
        for p in [self.SAVE_PATH, self.OUTPUT_PATH]:
            Path(p).mkdir(parents=True, exist_ok=True)

        ratio_sum = sum(self.SPLIT_RATIOS)
        if not np.isclose(ratio_sum, 1.0, atol=1e-6):
            raise ValueError(
                f"Config.SPLIT_RATIOS must sum to 1.0, got {self.SPLIT_RATIOS} "
                f"(sums to {ratio_sum}).")
        if len(self.SPLIT_RATIOS) != 3:
            raise ValueError(
                f"Config.SPLIT_RATIOS must have exactly 3 values, got "
                f"{len(self.SPLIT_RATIOS)}: {self.SPLIT_RATIOS}")

        if not (0.0 < self.CORESET_RATIO <= 1.0):
            raise ValueError(
                f"Config.CORESET_RATIO must be in (0, 1], got {self.CORESET_RATIO}")

        if self.DATA_ROOT == self._DATA_ROOT_PLACEHOLDER:
            raise ValueError(
                "Config.DATA_ROOT is still the default placeholder string. Set "
                "it to a real folder containing "
                f"{self.GOOD_DIRNAME!r} and {self.DEFECT_DIRNAME!r} subfolders, "
                "e.g. DATA_ROOT='/path/to/your/dataset'.\n"
                "แนะนำ: ให้ชี้ไปที่ DATA_ROOT เดียวกับ repo หลัก "
                "(Anomaly-Detection-THESIS) และตั้ง SPLIT_CACHE_PATH ให้ชี้ไปที่ "
                "splits/split_assignment.csv ไฟล์เดียวกัน เพื่อให้ train/val/test "
                "membership ตรงกันเป๊ะระหว่างสอง repo")
        if not Path(self.DATA_ROOT).is_dir():
            raise FileNotFoundError(
                f"Config.DATA_ROOT does not exist or is not a directory: "
                f"{self.DATA_ROOT!r}")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
