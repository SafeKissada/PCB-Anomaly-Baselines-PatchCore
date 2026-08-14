# PCB-Anomaly-Baselines

Repo แยกสำหรับรัน SOTA anomaly detection baseline (PatchCore, และในอนาคต
PaDiM / DRAEM / SimpleNet / RD4AD) บน dataset PCB defect/false-call เดียวกับ
[`Anomaly-Detection-THESIS`](https://github.com/SafeKissada/Anomaly-Detection-THESIS)
เพื่อเทียบกับ EXPERIMENT 0 (ConvNeXt frozen backbone + trainable autoencoder)
ในเล่ม thesis

**สถานะตอนนี้**: implement แล้ว 1/5 — **PatchCore**
ที่เหลือ (PaDiM, DRAEM, SimpleNet, RD4AD) ยังไม่ implement — โครง
`src/models/base.py` วางไว้ให้เพิ่มแบบเดียวกันได้เลย (ดูหัวข้อ "เพิ่ม
baseline ตัวถัดไป" ด้านล่าง)

## ทำไมต้องแยก repo

`Anomaly-Detection-THESIS` ผูกกับ reconstruction-based method
(autoencoder) ทั้ง data split logic บางส่วน (`_split_good_three_way`),
model, loss, และ evaluate.py's threshold diagnostic เฉพาะทาง — PatchCore
และ baseline อื่นๆ ไม่เทรน backbone เลย ไม่มี loss/epoch ให้ track จึงไม่
คุ้มจะยัดเข้า repo เดิม แต่ **`src/data/dataset.py` และ `src/evaluate.py`
ถูกคัดลอกมาแบบเป๊ะ (verbatim) จาก repo หลัก** เพื่อให้:

- Group-based split (ป้องกัน data leakage) เหมือนกันทุกตัวอักษร
- นิยาม metric (AUC, AP, Escape rate, Auto-clear rate ฯลฯ) เหมือนกันเป๊ะ
- **ถ้าตั้ง `SPLIT_CACHE_PATH` ให้ชี้ไปที่ไฟล์ `splits/split_assignment.csv`
  เดียวกับ repo หลัก** → train/val/test membership จะตรงกันทุกภาพ ทำให้
  ผลลัพธ์เทียบกันได้แบบ apples-to-apples ไม่มี confound เรื่อง data split

**อย่าลืมทำตามข้อสุดท้ายนี้** — ถ้าปล่อยให้แต่ละ repo คำนวณ split เอง (คนละ
random state เดินคนละแบบ แม้ seed เดียวกัน) ตัวเลขที่ได้จะเทียบกันไม่ได้
ตรงๆ อีกต่อไป

## โครงสร้าง

```
.
├── config/config.py          # Config dataclass (data path, backbone, coreset ratio, ...)
├── src/
│   ├── data/dataset.py        # คัดลอกจาก repo หลัก verbatim — ห้ามแก้ ให้ diff กันได้เสมอ
│   ├── evaluate.py             # คัดลอกจาก repo หลัก verbatim
│   ├── io_utils.py             # เซฟ final_results.json/scores.npz แบบเดียวกับ repo หลัก
│   └── models/
│       ├── base.py              # interface กลาง (fit/score) สำหรับทุก baseline
│       └── patchcore.py         # PatchCore implementation
├── scripts/run_patchcore.py   # entry point: fit -> threshold จาก val -> report บน test
├── RUN.py                     # แก้ config ตรงนี้แล้ว `python RUN.py`
└── tests/smoke_test.py        # รันได้แบบ offline (ไม่โหลด pretrained) เพื่อเช็ค pipeline ก่อนรันจริง
```

## วิธีรัน

```bash
pip install -r requirements.txt
```

แก้ `RUN.py`:

```python
OVERRIDES = dict(
    DATA_ROOT="/path/to/your/dataset",           # โฟลเดอร์ที่มี good/ กับ defect/
    SPLIT_CACHE_PATH="/path/to/Anomaly-Detection-THESIS/splits/split_assignment.csv",
    BACKBONE="wide_resnet50_2",                    # ตาม PatchCore paper ต้นฉบับ
    CORESET_RATIO=0.01,
    THRESHOLD_PERCENTILE=95.0,                      # ให้ตรงกับค่าที่ใช้ใน EXPERIMENT 0
)
```

```bash
python RUN.py
```

ผลลัพธ์ (`final_results_val.json`, `final_results_test.json`,
`scores_val.npz`, `scores_test.npz`) จะถูกเซฟไว้ใน `cfg.OUTPUT_PATH`

### รันทีละ group (เหมือน pipeline หลักที่ต้องรัน group 1–6 แยกกัน)

ตั้ง `DATA_ROOT` ให้ชี้เฉพาะ group นั้น (หรือกรองด้วย `GROUP_ID_REGEX` ถ้า
โครงสร้างโฟลเดอร์รวมทุก group ไว้ที่เดียว — ใช้ regex เดียวกับที่ตั้งไว้ใน
repo หลักถ้ามี)

### Smoke test ก่อนรันจริง (แนะนำ)

```bash
python tests/smoke_test.py
```

ใช้ภาพ dummy ขนาดเล็ก + backbone random-init (ไม่โหลด pretrained weight จาก
อินเทอร์เน็ต) รันจบใน ~1 วินาที เพื่อเช็คว่า pipeline ไม่มี error/shape
mismatch ก่อนไปรันกับ dataset จริงที่ใช้เวลานานกว่ามาก **ไม่ได้เช็ค
ความถูกต้องของตัวเลข metric** เพราะภาพ dummy ไม่มีสัญญาณ defect จริง

## ข้อควรระวังเฉพาะ dataset นี้ (สำคัญ — อ่านก่อนตีความผล)

1. **Memory bank สร้างจาก "Good" (=false call) เท่านั้น** ไม่ใช่ภาพงานดี
   ทั่วไปที่ไม่ถูก AOI flag เลย — ตาม `thesis_ai_context.md` นี่เป็น
   biased/narrow sample ของ "normal" ข้ออ้างว่า unsupervised generalize
   เข้า production ดีกว่ายังเป็นสมมติฐานที่ต้องพิสูจน์ ไม่ใช่ข้อสรุปสำเร็จรูป
   — PatchCore ก็มีความเสี่ยง distribution-shift แบบเดียวกัน ไม่ต่างจาก
   ConvNeXt+AE
2. **Group 3 และ 5 มี defect น้อยมาก** (41 และ 25 ภาพตามลำดับ) — เมื่อแบ่ง
   defect เป็น val/test (50/50 ตาม `_split_defect_two_way`) จะเหลือ test
   defect แค่ ~12-20 ภาพ ตัวเลข recall/escape rate จะมี variance สูง
   ตีความด้วยความระมัดระวัง อย่าฟันธงจากตัวเลขเดี่ยวๆ
3. **`CORESET_RATIO`** ค่า default 0.01 (1%) ตาม paper ต้นฉบับบน MVTec —
   dataset นี้มีจำนวนภาพ normal ต่อ group ต่างกันมาก (group 2: ~3,659 vs
   group 5: ~159) ควรลอง sensitivity ของค่านี้ต่อ group ที่ data น้อย
   แทนที่จะ fix ค่าเดียวข้ามทุก group
4. **`REWEIGHT_NUM_NEIGHBORS`** (ค่า default 9) ต้องไม่เกินขนาด memory bank
   — ลดอัตโนมัติถ้า memory bank เล็กกว่า (ดู `patchcore.py`) แต่ควรเช็คว่า
   ไม่เล็กจนกระทบคุณภาพ re-weighting โดยเฉพาะ group ที่ data น้อย

## เพิ่ม baseline ตัวถัดไป (PaDiM / DRAEM / SimpleNet / RD4AD)

1. สร้าง `src/models/<name>.py` implement `fit(normal_loader)` และ
   `score(loader) -> ScoreResult` ตาม `src/models/base.py`
2. สร้าง `scripts/run_<name>.py` โดย copy จาก `run_patchcore.py` แล้วสลับ
   import model
3. **ห้ามแก้ `src/data/dataset.py` หรือ `src/evaluate.py`** — ถ้าต้อง diff
   จาก repo หลักในอนาคต (เช่น repo หลักแก้ split logic) ต้อง sync ทั้งสอง
   repo พร้อมกันเสมอ ไม่งั้นผลจะเทียบกันไม่ได้อีกต่อไป

## Reference

Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, P.
(2022). *Towards Total Recall in Industrial Anomaly Detection.* CVPR 2022.
https://arxiv.org/abs/2106.08265
