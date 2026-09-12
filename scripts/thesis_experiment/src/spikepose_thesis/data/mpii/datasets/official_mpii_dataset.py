"""Official MPII train/valid protocol with HRNet-compatible preprocessing."""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from spikepose_thesis.data.mpii.core import MPII_FLIP_PAIRS, MPII_NUM_JOINTS
from spikepose_thesis.data.mpii.core.official_transforms import affine_matrix, transform_points
from spikepose_thesis.data.mpii.core.geometry import make_heatmaps


MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)
STD = np.asarray([0.229, 0.224, 0.225], np.float32)


class OfficialMPIIDataset(Dataset):
    """Read prepared official split JSONL without changing sample membership."""

    def __init__(self, metadata_path, images_dir, image_size=256, heatmap_size=64,
                 sigma=2.0, training=False, max_samples=None,
                 scale_factor=0.25, rotation_factor=30.0, flip=True,
                 color_rgb=True):
        self.metadata_path = Path(metadata_path)
        self.images_dir = Path(images_dir)
        self.image_size = int(image_size)
        self.heatmap_size = int(heatmap_size)
        self.sigma = float(sigma)
        self.training = bool(training)
        self.scale_factor = float(scale_factor)
        self.rotation_factor = float(rotation_factor)
        self.flip = bool(flip)
        self.color_rgb = bool(color_rgb)
        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"Official metadata not found: {self.metadata_path}. "
                "Run prepare_official_protocol.py first."
            )
        with self.metadata_path.open(encoding="utf-8") as handle:
            self.samples = [json.loads(line) for line in handle if line.strip()]
        if max_samples is not None:
            self.samples = self.samples[:int(max_samples)]
        if not self.samples:
            raise RuntimeError(f"No samples in {self.metadata_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        image_path = self.images_dir / item["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        # The original SimpleBaseline MPII release normalized OpenCV BGR
        # tensors, whereas the later HRNet release sets COLOR_RGB=true.
        if self.color_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Official annotations are MATLAB 1-based. The HRNet loader converts
        # joints and center to 0-based coordinates, shifts the crop down by
        # 15 scale units, and expands it by 1.25 before augmentation.
        joints = (np.asarray(item["keypoints"], np.float32)
                  .reshape(MPII_NUM_JOINTS, 2) - 1.0)
        visibility = np.asarray(item["visibility"], np.float32).copy()
        original_joints = joints.copy()
        scale = np.repeat(float(item["scale"]), 2).astype(np.float32)
        center = np.asarray(item["center"], np.float32).copy()
        if center[0] != -1:
            center[1] += 15.0 * scale[1]
            scale *= 1.25
        center -= 1.0
        rotation = 0.0
        if self.training:
            scale *= np.clip(
                np.random.randn() * self.scale_factor + 1.0,
                1.0 - self.scale_factor, 1.0 + self.scale_factor,
            )
            if random.random() <= 0.6:
                rotation = float(np.clip(
                    np.random.randn() * self.rotation_factor,
                    -2.0 * self.rotation_factor, 2.0 * self.rotation_factor,
                ))
            if self.flip and random.random() <= 0.5:
                image = image[:, ::-1].copy()
                joints[:, 0] = image.shape[1] - joints[:, 0] - 1
                center[0] = image.shape[1] - center[0] - 1
                for left, right in MPII_FLIP_PAIRS:
                    joints[[left, right]] = joints[[right, left]]
                    visibility[[left, right]] = visibility[[right, left]]
        matrix = affine_matrix(center, scale, rotation, (self.image_size, self.image_size))
        crop = cv2.warpAffine(
            image, matrix, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR,
        )
        transformed = transform_points(joints, matrix)
        heatmaps = make_heatmaps(
            transformed, visibility, self.image_size, self.heatmap_size, self.sigma,
        )
        tensor = torch.from_numpy(
            np.transpose((crop.astype(np.float32) / 255.0 - MEAN) / STD, (2, 0, 1)),
        ).float()
        inverse = affine_matrix(
            center, scale, rotation, (self.image_size, self.image_size), inverse=True,
        )
        custom_head_length = 0.75 * np.linalg.norm(original_joints[9] - original_joints[8])
        if not (visibility[8] > 0 and visibility[9] > 0):
            custom_head_length = np.nan
        return {
            "image": tensor,
            "heatmaps": torch.from_numpy(heatmaps),
            "keypoints": torch.from_numpy(transformed),
            "keypoints_original": torch.from_numpy(original_joints),
            "visibility": torch.from_numpy(visibility),
            "head_length": torch.tensor(float(item["head_length"]), dtype=torch.float32),
            "custom_head_length": torch.tensor(float(custom_head_length), dtype=torch.float32),
            "inverse_affine": torch.from_numpy(inverse.astype(np.float32)),
            # The generic training-time validator consumes ``inverse``.  Keep
            # the explicit official name as well for the final quarter-pixel
            # evaluation path.
            "inverse": torch.from_numpy(inverse.astype(np.float32)),
            "image_name": item["image"],
            "person_index": int(item.get("person_index", index)),
            "official_index": int(item["official_index"]),
        }
