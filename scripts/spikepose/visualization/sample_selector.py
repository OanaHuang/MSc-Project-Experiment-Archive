from __future__ import annotations

import json
import random
from pathlib import Path


def sample_id(dataset, index: int) -> str:
    if hasattr(dataset, "samples"):
        item = dataset.samples[index]
        image = str(item.get("image", index))
        person = item.get("person_index", 0)
        return f"{image}::{person}"
    return str(index)


def create_manifest(dataset, path: Path, seed: int = 2026,
                    first_count: int = 10, random_count: int = 10) -> dict:
    if len(dataset) < first_count + random_count:
        raise ValueError("The selected split must contain at least 20 samples")
    first = list(range(first_count))
    random_indices = random.Random(seed).sample(range(first_count, len(dataset)), random_count)
    manifest = {
        "selection_seed": seed,
        "first": [{"index": index, "id": sample_id(dataset, index)} for index in first],
        "random": [{"index": index, "id": sample_id(dataset, index)}
                   for index in random_indices],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
