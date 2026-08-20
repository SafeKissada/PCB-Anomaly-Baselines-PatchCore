"""
รัน PatchCore ซ้ำหลาย seed ต่อเนื่องกัน (multi-seed) — reuse OVERRIDES
เดียวกับ RUN.py ทุกประการ ไม่ copy dict ซ้ำ กัน 2 ไฟล์ไม่ sync กัน

auto-detect template จาก path ใน OVERRIDES ที่มี "SEED {n}" ฝังอยู่
แล้วแทนที่ด้วย seed จริงทุก key ที่อยู่ใน TEMPLATE_KEYS อัตโนมัติ —
logic เดียวกับ RUN_multi_seed.py ของ repo หลัก (Anomaly-Detection-THESIS)
ทุกประการ เพื่อให้ผลการทดลองเทียบกันได้โดยตรง

**ความแตกต่างสำคัญระหว่าง 2 กรณีของ multi-seed**:

กรณีที่ 1 — SPLIT_CACHE_PATH อยู่ใน TEMPLATE_KEYS (แยกต่อ seed):
  seed คุมทั้ง coreset randomness และ train/val/test split พร้อมกัน
  variance ที่วัดได้เป็นของทั้ง pipeline รวมกัน ไม่แยกแหล่ง

กรณีที่ 2 — SPLIT_CACHE_PATH ไม่อยู่ใน TEMPLATE_KEYS (แชร์ split เดียวกัน):
  seed คุมแค่ coreset randomness เท่านั้น (random projection direction +
  initial traversal point ใน greedy coreset subsampling)
  variance ที่วัดได้สะท้อน coreset randomness ล้วนๆ

ค่าปริยายของ TEMPLATE_KEYS รวม SPLIT_CACHE_PATH ไว้ด้วย (กรณีที่ 1)
ตรงกับที่ตัดสินใจไว้สำหรับ repo หลัก — ถ้าต้องการกรณีที่ 2 ให้เอา
'SPLIT_CACHE_PATH' ออกจาก TEMPLATE_KEYS ด้านล่าง

Two key cases for multi-seed:

Case 1 — SPLIT_CACHE_PATH in TEMPLATE_KEYS (separate split per seed):
  The seed governs both coreset randomness and train/val/test split.
  Measured variance is the combined variance of the whole pipeline,
  sources not separated.

Case 2 — SPLIT_CACHE_PATH NOT in TEMPLATE_KEYS (shared split):
  The seed governs coreset randomness only (random projection direction
  + initial traversal point in greedy coreset subsampling).
  Measured variance reflects coreset randomness alone.

TEMPLATE_KEYS defaults to including SPLIT_CACHE_PATH (Case 1), matching
the decision made for the main repo. Remove 'SPLIT_CACHE_PATH' from
TEMPLATE_KEYS below to switch to Case 2.
"""
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import RUN  # noqa: E402 — import หลัง sys.path.insert

from config.config import Config
from scripts.run_patchcore import run

# ── seed ที่จะรัน ────────────────────────────────────────────────────
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# ── key ที่ต้องแยกตาม seed (ต้องมี "SEED {n}" ฝังอยู่ใน value) ────────
# ค่าปริยาย: แยกทั้ง SPLIT_CACHE_PATH, SAVE_PATH, OUTPUT_PATH ตาม seed
# (กรณีที่ 1 — seed คุมทั้ง split และ coreset randomness พร้อมกัน)
# ลบ 'SPLIT_CACHE_PATH' ออกถ้าต้องการแชร์ split เดียวกันทุก seed แทน
#
# Default: split all 3 paths per seed (Case 1 — seed governs both split
# and coreset randomness). Remove 'SPLIT_CACHE_PATH' to share one split
# across all seeds instead (Case 2).
TEMPLATE_KEYS = ['SPLIT_CACHE_PATH', 'SAVE_PATH', 'OUTPUT_PATH']

# ── auto-detect template จาก OVERRIDES ปัจจุบัน ─────────────────────
# อ่าน SEED ปัจจุบัน (เช่น 42 ถ้า OVERRIDES['SEED']=42) แล้วหา marker
# "SEED 42" ใน path ทุก key ใน TEMPLATE_KEYS — fail fast ถ้าไม่เจอ
# เพื่อป้องกัน silent path mismatch (เช่น SAVE_PATH ชี้ผิดโฟลเดอร์
# โดยไม่มี error) เหมือน fail-fast ของ repo หลัก
#
# Reads the current SEED (e.g. 42 if OVERRIDES['SEED']=42) and looks
# for marker "SEED 42" in every path in TEMPLATE_KEYS — fails fast if
# not found to prevent a silent path mismatch (e.g. SAVE_PATH pointing
# at the wrong folder with no error), same as the main repo's fail-fast.
_current_seed = RUN.OVERRIDES['SEED']
_marker = f'SEED {_current_seed}'
_placeholder = 'SEED {seed}'

path_templates = {}
for key in TEMPLATE_KEYS:
    original_value = RUN.OVERRIDES[key]
    if _marker not in original_value:
        raise ValueError(
            f"ไม่เจอ '{_marker}' ใน RUN.OVERRIDES['{key}'] "
            f"(= {original_value!r}) — ต้องมีข้อความ '{_marker}' อยู่ใน "
            f"path นั้นให้ script แทนที่ด้วย seed อื่นได้ ถ้าตั้งชื่อ "
            f"โฟลเดอร์ต่างจากนี้ (เช่น เว้นวรรคไม่เหมือนกัน) ให้แก้ให้ตรง "
            f"ก่อน หรือแก้ TEMPLATE_KEYS ด้านบนของไฟล์นี้เอง\n"
            f"/ '{_marker}' not found in RUN.OVERRIDES['{key}'] "
            f"(= {original_value!r}). The path must contain the exact "
            f"text '{_marker}' for this script to substitute other seeds "
            f"in. Fix the folder name (e.g. inconsistent spacing) first, "
            f"or edit TEMPLATE_KEYS at the top of this file.")
    path_templates[key] = original_value.replace(_marker, _placeholder)

print("Path templates ที่ตรวจพบ (จะแทน {seed} ด้วยเลข seed แต่ละรอบ):")
for key, tmpl in path_templates.items():
    print(f"  {key} = {tmpl!r}")

results_log = []

for i, seed in enumerate(SEEDS, start=1):
    print(f"\n{'=' * 70}")
    print(f" MULTI-SEED RUN [{i}/{len(SEEDS)}] — SEED={seed}")
    print(f"{'=' * 70}")

    # แก้ OVERRIDES ตรงๆ — Config() ที่ถูกสร้างใน run() อ่านค่าจาก
    # RUN.OVERRIDES สดทุกครั้ง ทำให้ seed แต่ละรอบได้ path ถูกต้อง
    # โดยไม่ต้องส่งผ่าน argument เพิ่ม — pattern เดียวกับ repo หลัก
    #
    # Mutate OVERRIDES directly — Config() created inside run() reads
    # from RUN.OVERRIDES fresh every time, so each seed gets the correct
    # path without needing extra arguments. Same pattern as the main repo.
    RUN.OVERRIDES['SEED'] = seed
    for key, tmpl in path_templates.items():
        RUN.OVERRIDES[key] = tmpl.format(seed=seed)

    print(f"  SPLIT_CACHE_PATH -> {RUN.OVERRIDES['SPLIT_CACHE_PATH']}")
    print(f"  SAVE_PATH        -> {RUN.OVERRIDES['SAVE_PATH']}")
    print(f"  OUTPUT_PATH      -> {RUN.OVERRIDES['OUTPUT_PATH']}")

    try:
        cfg = Config(**RUN.OVERRIDES)
        run(cfg)
        results_log.append((seed, 'OK', None))
        print(f"\n✅ seed={seed} เสร็จสมบูรณ์ -> {RUN.OVERRIDES['SAVE_PATH']}")

    except Exception as e:
        # seed ที่พังกลางคัน ไม่ทำให้เสียงานของ seed ก่อนหน้าที่รันสำเร็จ
        # ไปแล้ว — log ไว้แล้วไปต่อ seed ถัดไป (เช่นเดียวกับ repo หลัก)
        #
        # A seed crashing midway does not destroy the work of previously
        # successful seeds — log it and move on (same as the main repo).
        results_log.append((seed, 'FAILED', str(e)))
        print(f"\n❌ seed={seed} ล้มเหลว: {e}")
        traceback.print_exc()
        print("ข้าม seed นี้ ไปทำ seed ถัดไปต่อ...")
        continue

# ── สรุปผลรวมท้ายสุด ───────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(" สรุปผล Multi-Seed Run")
print(f"{'=' * 70}")
for seed, status, err in results_log:
    line = f"  seed={seed:<4}  {status}"
    if err:
        line += f"  ({err})"
    print(line)

n_ok = sum(1 for _, s, _ in results_log if s == 'OK')
print(f"\nสำเร็จ {n_ok}/{len(SEEDS)} seed")
if n_ok < len(SEEDS):
    print(
        "⚠️  มี seed ที่ล้มเหลว — เช็ค traceback ด้านบนก่อนเอาผลไปสรุปสถิติ "
        "(mean_auc/std_auc ต้องคำนวณจากเฉพาะ seed ที่ OK เท่านั้น)"
    )