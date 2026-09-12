import json

import pytest

from spikepose_thesis.data.ntu.preflight import (
    PREFLIGHT_SCHEMA_VERSION,
    load_validated_sample_ids,
    metadata_fingerprint,
    metadata_key,
)


def test_pose_manifest_is_reused_only_while_metadata_matches(tmp_path):
    metadata = tmp_path / "train.csv"
    metadata.write_text("sample_id\nS001C001P001R001A001\n", encoding="utf-8")
    manifest_path = tmp_path / "pose_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "ready": True,
        "metadata": {
            metadata_key(metadata): {"sha256": metadata_fingerprint(metadata)},
        },
        "valid_sample_ids": ["S001C001P001R001A001"],
    }), encoding="utf-8")
    data = {"pose_validation_manifest": str(manifest_path)}

    assert load_validated_sample_ids(data, metadata) == {
        "S001C001P001R001A001",
    }

    metadata.write_text(
        "sample_id\nS001C001P001R001A001\nS002C001P001R001A001\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not match"):
        load_validated_sample_ids(data, metadata)
