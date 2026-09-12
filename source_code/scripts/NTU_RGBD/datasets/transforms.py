from __future__ import annotations

import cv2
import numpy as np
import torch

from scripts.NTU_RGBD.core.config import NTU_FLIP_PAIRS


IMAGENET_MEAN = np.asarray(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.asarray(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


class PoseTransform:
    def __init__(
        self,
        image_size: int = 224,
        training: bool = False,
        scale_range: tuple[float, float] | None = None,
        rotation_degrees: float = 0.0,
        flip_probability: float = 0.0,
        flip_pairs: tuple[tuple[int, int], ...] = NTU_FLIP_PAIRS,
    ) -> None:
        self.image_size = int(image_size)
        self.training = bool(training)
        self.scale_range = (None if scale_range is None else
                            tuple(float(value) for value in scale_range))
        self.rotation_degrees = float(rotation_degrees)
        self.flip_probability = float(flip_probability)
        self.flip_pairs = tuple((int(left), int(right)) for left, right in flip_pairs)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.scale_range is not None and (
            len(self.scale_range) != 2 or self.scale_range[0] <= 0
            or self.scale_range[0] > self.scale_range[1]
        ):
            raise ValueError("scale_range must contain two positive ordered values")
        if self.rotation_degrees < 0:
            raise ValueError("rotation_degrees must be non-negative")
        if not 0.0 <= self.flip_probability <= 1.0:
            raise ValueError("flip_probability must be in [0, 1]")

    def sample_parameters(self) -> dict[str, float | bool]:
        """Sample one augmentation shared by every frame in a clip."""
        scale = (np.random.uniform(*self.scale_range)
                 if self.training and self.scale_range is not None else 1.0)
        angle = (np.random.uniform(-self.rotation_degrees, self.rotation_degrees)
                 if self.training and self.rotation_degrees > 0 else 0.0)
        flip = bool(
            self.training and self.flip_probability > 0
            and np.random.random() < self.flip_probability
        )
        return {"scale": float(scale), "angle": float(angle), "flip": flip}

    def __call__(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        visibility: np.ndarray,
        parameters: dict[str, float | bool] | None = None,
    ) -> dict[str, torch.Tensor]:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")

        keypoints = np.asarray(
            keypoints,
            dtype=np.float32,
        ).copy()

        visibility = np.asarray(
            visibility,
            dtype=np.float32,
        ).copy()

        original_height, original_width = image.shape[:2]

        if (original_width, original_height) != (self.image_size, self.image_size):
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )

        keypoints[:, 0] *= (
            self.image_size / original_width
        )

        keypoints[:, 1] *= (
            self.image_size / original_height
        )

        parameters = self.sample_parameters() if parameters is None else parameters
        scale = float(parameters.get("scale", 1.0))
        angle = float(parameters.get("angle", 0.0))
        flip = bool(parameters.get("flip", False))

        if self.training and (scale != 1.0 or angle != 0.0):
            center = ((self.image_size - 1) * 0.5, (self.image_size - 1) * 0.5)
            matrix = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)
            image = cv2.warpAffine(
                image, matrix, (self.image_size, self.image_size),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
            )
            homogeneous = np.concatenate(
                (keypoints, np.ones((len(keypoints), 1), dtype=np.float32)), axis=1,
            )
            keypoints = homogeneous @ matrix.T

        if self.training and flip:
            image = image[:, ::-1].copy()
            keypoints[:, 0] = self.image_size - 1 - keypoints[:, 0]
            for left, right in self.flip_pairs:
                keypoints[[left, right]] = keypoints[[right, left]]
                visibility[[left, right]] = visibility[[right, left]]

        in_bounds = (
            np.isfinite(keypoints).all(axis=1)
            & (keypoints[:, 0] >= 0) & (keypoints[:, 0] < self.image_size)
            & (keypoints[:, 1] >= 0) & (keypoints[:, 1] < self.image_size)
        )
        visibility *= in_bounds.astype(np.float32)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = image.astype(
            np.float32
        ) / 255.0

        image = (
            image - IMAGENET_MEAN
        ) / IMAGENET_STD

        image = np.transpose(
            image,
            (2, 0, 1),
        )

        return {
            "image": torch.from_numpy(
                image
            ).float(),

            "keypoints": torch.from_numpy(
                keypoints
            ).float(),

            "visibility": torch.from_numpy(
                visibility
            ).float(),
        }


def build_train_transform(
    image_size: int = 224,
    scale_range: tuple[float, float] | None = None,
    rotation_degrees: float = 0.0,
    flip_probability: float = 0.0,
) -> PoseTransform:
    return PoseTransform(
        image_size=image_size,
        training=True,
        scale_range=scale_range,
        rotation_degrees=rotation_degrees,
        flip_probability=flip_probability,
    )


def build_eval_transform(
    image_size: int = 224,
) -> PoseTransform:
    return PoseTransform(
        image_size=image_size,
        training=False,
    )
