"""
PatchCore (Roth et al., "Towards Total Recall in Industrial Anomaly
Detection", CVPR 2022) — https://arxiv.org/abs/2106.08265

สรุป pipeline:
  1. ดึง patch feature จาก mid-level layer ของ ImageNet-pretrained CNN
     (layer2 + layer3 ของ ResNet-family ตาม paper ต้นฉบับ) แบบ frozen
     ไม่เทรน backbone เลย (ต่างจาก repo หลักที่เทรน autoencoder)
  2. Locally-aware patch feature: average-pool รอบ patch (kernel=3) ให้
     แต่ละ patch feature มี context จาก neighbor รอบข้างด้วย
  3. สร้าง memory bank จาก patch feature ของภาพ "normal" (good/false-call)
     ทั้งหมดใน train split แล้วบีบด้วย greedy coreset subsampling (k-center)
     ให้เหลือ CORESET_RATIO ของ patch ทั้งหมด (ปกติ ~1%) เพื่อให้ inference
     เร็วพอจะใช้งานจริง
  4. Inference: patch score = ระยะห่างไปยัง nearest neighbor ใน memory bank
     Image score = ระยะของ patch ที่แย่ที่สุด คูณด้วย re-weighting factor
     (ดูเหตุผลใน _reweight_image_score)

หมายเหตุสำคัญสำหรับ dataset นี้ (ต่างจาก MVTec ที่ paper ต้นฉบับ benchmark):
  - Memory bank สร้างจาก "Good" (=false call) เท่านั้น ซึ่งตาม context เป็น
    biased/narrow sample ของ normal ไม่ใช่ภาพงานดีทั่วไปที่ไม่ถูก AOI flag
    เลย — ตีความผลลัพธ์โดยเผื่อ distribution-shift นี้ไว้เสมอ (ดู
    thesis_ai_context.md หัวข้อ "สิ่งที่ยังเป็นสมมติฐาน")
  - Escape rate ยังเป็น metric ที่ต้อง track คู่กับ AUROC/AP เหมือน
    experiment 0 ของ repo หลัก เพราะเป็นตัวชี้ deployability จริง ไม่ใช่แค่
    ความสามารถแยก class เชิงสถิติ
"""
import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

logger = logging.getLogger("PatchCore")

_BACKBONE_FACTORY = {
    "wide_resnet50_2": (torchvision.models.wide_resnet50_2,
                         torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V2),
    "resnet18": (torchvision.models.resnet18,
                 torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (torchvision.models.resnet50,
                 torchvision.models.ResNet50_Weights.IMAGENET1K_V2),
}


class _FeatureExtractor(torch.nn.Module):
    """Frozen backbone + forward hooks บน cfg.FEATURE_LAYERS (ปกติ layer2, layer3).
    ไม่มี trainable parameter เลยสักตัว — ต่างจาก autoencoder ใน repo หลักที่
    ตัว backbone frozen แต่มี autoencoder ที่ต้องเทรน ส่วน PatchCore ทั้งโมเดล
    ไม่ต้องเทรนอะไรเลยนอกจาก build memory bank
    """

    def __init__(self, backbone_name: str, layers: Tuple[str, ...], device,
                 pretrained: bool = True):
        super().__init__()
        if backbone_name not in _BACKBONE_FACTORY:
            raise ValueError(
                f"Unknown BACKBONE {backbone_name!r}. Supported: "
                f"{list(_BACKBONE_FACTORY)}")
        ctor, weights = _BACKBONE_FACTORY[backbone_name]
        if not pretrained:
            logger.warning(
                "pretrained=False: ใช้ random-init weights — สำหรับ smoke "
                "test/offline dev เท่านั้น ห้ามใช้รันผลจริงเด็ดขาด เพราะ "
                "PatchCore พึ่ง ImageNet feature ทั้งหมด ไม่มีการเทรนเพิ่ม")
        self.backbone = ctor(weights=weights if pretrained else None)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.to(device)

        self.layers = layers
        self._features = {}
        self._hooks = []
        for name in layers:
            module = dict(self.backbone.named_modules())[name]
            self._hooks.append(
                module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(_module, _input, output):
            self._features[name] = output
        return hook

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> "list[torch.Tensor]":
        self._features = {}
        self.backbone(x)
        return [self._features[name] for name in self.layers]

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def _locally_aware_patchify(feat: torch.Tensor, kernel: int) -> torch.Tensor:
    """Average-pool รอบแต่ละ patch (stride=1, padding เพื่อรักษาขนาดเดิม)
    เพื่อให้ patch feature มี local context ตาม PatchCore paper section 3.1
    """
    pad = kernel // 2
    return F.avg_pool2d(feat, kernel_size=kernel, stride=1, padding=pad)


def _embed_batch(extractor: _FeatureExtractor, images: torch.Tensor,
                  patch_pool_kernel: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """คืน patch embeddings [B*H*W, C] ของ batch หนึ่ง + spatial shape (H, W)
    ที่ resolution ของ feature map แรก (layer2) — layer อื่นถูก resize ให้
    เท่ากันก่อน concat ตาม channel
    """
    feats = extractor(images)  # list of [B, C_l, H_l, W_l], ละเอียดลดลงตาม layer ที่ลึกขึ้น
    feats = [_locally_aware_patchify(f, patch_pool_kernel) for f in feats]

    ref_h, ref_w = feats[0].shape[-2:]
    resized = [feats[0]]
    for f in feats[1:]:
        resized.append(F.interpolate(f, size=(ref_h, ref_w), mode="bilinear",
                                      align_corners=False))
    embedding = torch.cat(resized, dim=1)  # [B, C_total, H, W]

    B, C, H, W = embedding.shape
    patches = embedding.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [B*H*W, C]
    return patches, (H, W)


def _greedy_coreset_subsample(features: torch.Tensor, target_size: int,
                               projection_dim: int, device) -> torch.Tensor:
    """Greedy k-center (minimax facility location) coreset subsampling —
    PatchCore paper Algorithm 1. เลือก subset ที่ "ครอบคลุม" feature space
    เดิมได้ดีที่สุดในแง่ maximum-distance-to-nearest-selected-point แทนที่จะ
    random sample ธรรมดา (ซึ่งจะทิ้ง rare-but-important normal pattern ไปได้ง่าย)

    ทำ random projection ลงมิติต่ำก่อน (Johnson-Lindenstrauss) เพื่อให้การหา
    farthest point แต่ละรอบเร็วขึ้น โดยไม่กระทบ relative distance มากนัก
    """
    n = features.shape[0]
    if target_size >= n:
        return features

    torch.manual_seed(0)
    proj_dim = min(projection_dim, features.shape[1])
    R = torch.randn(features.shape[1], proj_dim, device=device) / np.sqrt(proj_dim)
    projected = features @ R  # [n, proj_dim]

    selected_idx = [int(torch.randint(0, n, (1,)).item())]
    min_dist = torch.cdist(projected, projected[selected_idx]).squeeze(1)  # [n]

    for _ in range(target_size - 1):
        next_idx = int(torch.argmax(min_dist).item())
        selected_idx.append(next_idx)
        new_dist = torch.cdist(projected, projected[[next_idx]]).squeeze(1)
        min_dist = torch.minimum(min_dist, new_dist)
        min_dist[next_idx] = -1  # กันไม่ให้เลือกซ้ำ

    return features[selected_idx]


class PatchCore:
    """ใช้ตาม interface ของ BaseAnomalyModel (fit / score) แต่ไม่ inherit
    ตรงๆ เพื่อเลี่ยง import cycle กับ base.py — ดู scripts/run_patchcore.py
    ว่าประกอบเข้ากับ ScoreResult ยังไง
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.DEVICE
        self.extractor = _FeatureExtractor(
            cfg.BACKBONE, cfg.FEATURE_LAYERS, self.device,
            pretrained=getattr(cfg, "PRETRAINED", True))
        self.memory_bank: torch.Tensor = None  # [M, C], ตั้งค่าตอน fit()
        self.embed_spatial_shape: Tuple[int, int] = None

    @torch.no_grad()
    def fit(self, normal_loader) -> None:
        logger.info("PatchCore.fit(): extracting patch features จากภาพ normal ทั้งหมด...")
        all_patches = []
        for batch in normal_loader:
            images = batch[0].to(self.device)  # AnomalyDataset[0] = normalized_tensor
            patches, shape = _embed_batch(
                self.extractor, images, self.cfg.PATCH_POOL_KERNEL)
            self.embed_spatial_shape = shape
            all_patches.append(patches.cpu())

        all_patches = torch.cat(all_patches, dim=0).to(self.device)
        n_total = all_patches.shape[0]
        target_size = max(1, int(n_total * self.cfg.CORESET_RATIO))
        logger.info(f"รวม patch ทั้งหมด {n_total:,} — coreset subsampling เหลือ "
                    f"{target_size:,} ({self.cfg.CORESET_RATIO:.1%})")

        self.memory_bank = _greedy_coreset_subsample(
            all_patches, target_size, self.cfg.CORESET_PROJECTION_DIM, self.device)
        logger.info(f"Memory bank พร้อมใช้: {self.memory_bank.shape}")

    @torch.no_grad()
    def _knn_min_dist(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """หา nearest-neighbor distance + index ใน memory bank สำหรับแต่ละ
        patch, ทำแบบ chunked กัน OOM เมื่อ memory bank ใหญ่
        """
        n = patches.shape[0]
        min_dists = torch.empty(n, device=self.device)
        min_idx = torch.empty(n, dtype=torch.long, device=self.device)
        chunk = self.cfg.KNN_CHUNK_SIZE
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            d = torch.cdist(patches[start:end], self.memory_bank)  # [chunk, M]
            vals, idx = d.min(dim=1)
            min_dists[start:end] = vals
            min_idx[start:end] = idx
        return min_dists, min_idx

    @torch.no_grad()
    def _reweight_image_score(self, patch_dists: torch.Tensor,
                               patch_nn_idx: torch.Tensor,
                               worst_patch_feature: torch.Tensor) -> float:
        """PatchCore paper eq. (7): re-weight ค่า max patch distance ด้วยว่า
        nearest-neighbor ของ patch ที่แย่ที่สุด (m*) นั้น "โดดเดี่ยว" ใน
        memory bank แค่ไหน — ถ้า m* มีเพื่อนบ้านใกล้ๆ ใน memory bank เยอะที่
        ก็ใกล้ worst patch feature ด้วย แปลว่าบริเวณนั้นของ feature space มี
        normal coverage เยอะอยู่แล้ว ความผิดปกติจริงน่าจะต่ำกว่าที่ raw
        distance บอก — ในทางกลับกัน ถ้า m* โดดเดี่ยว ให้เชื่อ raw distance เต็มๆ
        """
        s_star = float(patch_dists.max())
        worst_idx = int(patch_dists.argmax())
        m_star_idx = int(patch_nn_idx[worst_idx])
        m_star = self.memory_bank[m_star_idx : m_star_idx + 1]  # [1, C]

        b = min(self.cfg.REWEIGHT_NUM_NEIGHBORS, self.memory_bank.shape[0] - 1)
        if b <= 0:
            return s_star

        d_to_bank = torch.cdist(m_star, self.memory_bank).squeeze(0)  # [M]
        d_to_bank[m_star_idx] = float("inf")  # กัน m* จับคู่ตัวเอง
        neighbor_idx = torch.topk(d_to_bank, k=b, largest=False).indices  # [b]

        f_star = worst_patch_feature.unsqueeze(0)  # [1, C]
        neighbor_feats = self.memory_bank[neighbor_idx]  # [b, C]
        d_f_to_neighbors = torch.cdist(f_star, neighbor_feats).squeeze(0)  # [b]

        weight = 1.0 - (torch.exp(torch.tensor(s_star, device=self.device))
                         / torch.exp(d_f_to_neighbors).sum())
        weight = float(torch.clamp(weight, min=0.0, max=1.0))
        return weight * s_star

    @torch.no_grad()
    def score(self, loader):
        from src.models.base import ScoreResult  # lazy import กัน circular import

        if self.memory_bank is None:
            raise RuntimeError("PatchCore.score() ถูกเรียกก่อน fit() — ต้อง "
                                "build memory bank จากภาพ normal ก่อนเสมอ")

        image_scores, labels, paths = [], [], []
        pixel_maps = []
        H, W = self.embed_spatial_shape

        for batch in loader:
            images, _orig, _preproc, batch_paths, batch_labels, _size = batch
            images = images.to(self.device)
            B = images.shape[0]

            patches, (h, w) = _embed_batch(
                self.extractor, images, self.cfg.PATCH_POOL_KERNEL)
            assert (h, w) == (H, W), (
                f"Spatial shape เปลี่ยนระหว่าง fit() ({H},{W}) กับ score() "
                f"({h},{w}) — เช็คว่า IMAGE_SIZE ตรงกันทั้งสองรอบ")

            dists, nn_idx = self._knn_min_dist(patches)  # [B*H*W]
            dists_per_img = dists.view(B, H * W)
            nn_idx_per_img = nn_idx.view(B, H * W)
            patches_per_img = patches.view(B, H * W, -1)

            for i in range(B):
                score = self._reweight_image_score(
                    dists_per_img[i], nn_idx_per_img[i],
                    patches_per_img[i, int(dists_per_img[i].argmax())])
                image_scores.append(score)

                pmap = dists_per_img[i].view(1, 1, H, W)
                pmap = F.interpolate(pmap, size=self.cfg.IMAGE_SIZE,
                                      mode="bilinear", align_corners=False)
                pmap = _gaussian_smooth(pmap.squeeze().cpu().numpy(),
                                         self.cfg.HEATMAP_SIGMA)
                pixel_maps.append(pmap)

            labels.extend([0 if lb == "normal" else 1 for lb in batch_labels])
            paths.extend(batch_paths)

        return ScoreResult(
            image_scores=np.array(image_scores, dtype=np.float64),
            labels=np.array(labels, dtype=np.int64),
            paths=paths,
            pixel_maps=np.stack(pixel_maps, axis=0),
        )


def _gaussian_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr, sigma=sigma)
