from scripts.NTU_RGBD.train_ntu_gpu_queue import F_BATCH, G_BATCH, QUEUE, build_command
from scripts.spikepose.experiments import experiment_roots, resolve_config, validate_config
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def resolved(experiment):
    value = resolve_config(
        experiment_roots(ROOT, "ntu_rgbd"), experiment,
        ROOT / "scripts/NTU_RGBD/configs/task.yaml",
        ROOT / "scripts/NTU_RGBD/configs/training.yaml",
    )
    validate_config(value)
    return value


def test_queue_has_three_complete_four_gpu_batches():
    assert len(QUEUE) == 3
    assert all(len(batch) == 4 for batch in QUEUE)
    assert all({run.gpu for run in batch} == {0, 1, 2, 3} for batch in QUEUE)


def test_frame_count_configs_match_their_names():
    for steps, run in enumerate(F_BATCH, 1):
        config = resolved(run.experiment)
        assert config["model"]["num_steps"] == steps
        assert config["model"]["temporal"]["input_strategy"] == "frames"


def test_gap_batch_and_twenty_epoch_commands():
    assert [(run.display_id, run.gap) for run in G_BATCH] == [
        ("G2", 2), ("G3", 3), ("G5", 5), ("R2", 1),
    ]
    for batch in QUEUE[1:]:
        for run in batch:
            command = build_command(run)
            assert command[command.index("--epochs") + 1] == "20"
            assert command[command.index("--device") + 1] == f"cuda:{run.gpu}"
            assert command[command.index("--temporal-frame-gap") + 1] == str(run.gap)
