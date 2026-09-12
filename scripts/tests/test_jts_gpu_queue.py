from scripts.NTU_RGBD.train_jts_gpu_queue import (
    HAND_JOINTS, RUNS, build_command,
)


def test_jts_gpu_queue_maps_one_ranked_run_per_gpu():
    assert [run.experiment for run in RUNS] == ["jts1", "jts2", "jts3", "jts4"]
    assert [run.gpu for run in RUNS] == [0, 1, 2, 3]
    assert [run.stage_steps for run in RUNS] == [
        "4-4-4-4", "4-3-2-1", "4-4-3-1", "4-4-4-1",
    ]
    for run in RUNS:
        command = build_command(run)
        assert command[command.index("--experiment") + 1] == run.experiment
        assert command[command.index("--device") + 1] == f"cuda:{run.gpu}"
        assert command[command.index("--physical-gpu") + 1] == str(run.gpu)
        assert "--skip-visualization" in command


def test_jts_queue_hand_group_contains_fine_distal_joints():
    assert HAND_JOINTS == (
        "wrist_left", "wrist_right", "hand_left", "hand_right",
        "hand_tip_left", "hand_tip_right", "thumb_left", "thumb_right",
    )
