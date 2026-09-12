from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from spikepose_thesis.data.mpii.core import MPII_FLIP_PAIRS, MPII_NUM_JOINTS
from spikepose_thesis.data.mpii.core.geometry import crop_person, make_heatmaps

MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)
STD = np.asarray([0.229, 0.224, 0.225], np.float32)


class MPIIPoseDataset(Dataset):
    """Person-centric MPII samples produced by 01_Prepare_MPII_Metadata.py."""

    def __init__(self, metadata_path, images_dir, image_size=224, heatmap_size=56, sigma=2.0,
                 crop_expansion=1.25, training=False, max_samples=None,
                 scale_range=None, rotation_degrees=0.0):
        self.metadata_path = Path(metadata_path)
        self.images_dir = Path(images_dir)
        self.image_size = int(image_size)
        self.heatmap_size = int(heatmap_size)
        self.sigma = float(sigma)
        self.crop_expansion = float(crop_expansion)
        self.training = bool(training)
        self.scale_range = (None if scale_range is None else
                            tuple(float(value) for value in scale_range))
        self.rotation_degrees = float(rotation_degrees)
        if self.scale_range is not None:
            if len(self.scale_range) != 2 or not 0 < self.scale_range[0] <= self.scale_range[1]:
                raise ValueError("scale_range must contain two positive ordered values")
        if self.rotation_degrees < 0:
            raise ValueError("rotation_degrees must be non-negative")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}. Run 01_Prepare_MPII_Metadata.py")
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
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        keypoints = np.asarray(item["keypoints"], np.float32).reshape(MPII_NUM_JOINTS, 2)
        visibility = np.asarray(item["visibility"], np.float32)
        crop, keypoints_crop, box, inverse = crop_person(
            image, keypoints, item["center"], item["scale"], self.image_size, self.crop_expansion)
        if self.training and (self.scale_range is not None or self.rotation_degrees > 0):
            scale = (np.random.uniform(*self.scale_range)
                     if self.scale_range is not None else 1.0)
            angle = (np.random.uniform(-self.rotation_degrees, self.rotation_degrees)
                     if self.rotation_degrees > 0 else 0.0)
            center = ((self.image_size - 1) * 0.5, (self.image_size - 1) * 0.5)
            matrix = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)
            crop = cv2.warpAffine(
                crop, matrix, (self.image_size, self.image_size),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
            )
            homogeneous = np.concatenate(
                (keypoints_crop, np.ones((len(keypoints_crop), 1), np.float32)), axis=1,
            )
            keypoints_crop = homogeneous @ matrix.T
        if self.training and np.random.random() < 0.5:
            crop = crop[:, ::-1].copy()
            keypoints_crop[:, 0] = self.image_size - 1 - keypoints_crop[:, 0]
            for left, right in MPII_FLIP_PAIRS:
                keypoints_crop[[left, right]] = keypoints_crop[[right, left]]
                visibility[[left, right]] = visibility[[right, left]]
        in_bounds = ((keypoints_crop[:, 0] >= 0) & (keypoints_crop[:, 0] < self.image_size)
                     & (keypoints_crop[:, 1] >= 0) & (keypoints_crop[:, 1] < self.image_size))
        visibility *= in_bounds.astype(np.float32)
        heatmaps = make_heatmaps(keypoints_crop, visibility, self.image_size, self.heatmap_size, self.sigma)
        custom_head_length = np.nan
        if visibility[8] > 0 and visibility[9] > 0:
            value = 1.0 * np.linalg.norm(keypoints[8] - keypoints[9])
            if np.isfinite(value) and value > 0:
                custom_head_length = float(value)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose((rgb - MEAN) / STD, (2, 0, 1))).float()
        return {
            "image": tensor,
            "heatmaps": torch.from_numpy(heatmaps),
            "keypoints": torch.from_numpy(keypoints_crop),
            "keypoints_original": torch.from_numpy(keypoints),
            "visibility": torch.from_numpy(visibility),
            "head_length": torch.tensor(float(item["head_length"]), dtype=torch.float32),
            "custom_head_length": torch.tensor(custom_head_length, dtype=torch.float32),
            "bbox": torch.from_numpy(box),
            "inverse": torch.from_numpy(inverse),
            "image_name": item["image"],
            "person_index": int(item["person_index"]),
        }
