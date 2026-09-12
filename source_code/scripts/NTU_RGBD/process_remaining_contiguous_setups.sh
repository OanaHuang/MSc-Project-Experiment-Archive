#!/usr/bin/env bash
set -euo pipefail

project=/extra2/yunhao/MSc_Project
dataset="$project/Datasets/NTU_RGBD"
archives=/extra2/yunhao/ntu60_archives
python="$project/.venv/bin/python"
default_setups=(S003 S004 S005 S006 S007 S008 S009 S011 S012 S013 S014 S015 S016 S017)
if [ "$#" -gt 0 ]; then
    setups=("$@")
else
    setups=("${default_setups[@]}")
fi

[ "$(realpath "$project")" = "$project" ]
[ "$(realpath "$dataset")" = "$dataset" ]
[ -x "$python" ]

cd "$project"
for setup in "${setups[@]}"; do
    archive="$archives/nturgbd_rgb_${setup,,}.zip"
    raw="$dataset/rgb_videos/$setup"
    video_root="$raw/nturgb+d_rgb"
    output="$dataset/frames/$setup/contiguous_2x16"
    printf '[setup] %s starting\n' "$setup"
    [ -f "$archive" ] && [ ! -L "$archive" ]
    if [ -e "$raw" ]; then
        [ -d "$raw" ] && [ ! -L "$raw" ]
        [ "$(realpath "$raw")" = "$raw" ]
    fi

    "$python" scripts/NTU_RGBD/prepare_setup_frames.py \
        --setup "$setup" --archive "$archive" --dataset-root "$dataset" \
        --stage unpack
    "$python" scripts/NTU_RGBD/extract_contiguous_video_frames.py \
        --setup "$setup" --video-root "$video_root" --output-root "$output" \
        --clip-length 16 --clips-per-video 2 --jpeg-quality 90 --workers 8

    expected=$(
        "$python" -c \
            'import json,sys; p=json.load(open(sys.argv[1])); assert p["setup"]==sys.argv[2]; assert p["frames"]==p["videos"]*p["frames_per_video"]; print(p["frames"])' \
            "$output/manifest.json" "$setup"
    )
    actual=$(find "$output" -type f -name '*.jpg' | wc -l)
    [ "$actual" -eq "$expected" ]
    [ -d "$raw" ] && [ ! -L "$raw" ]
    [ "$(realpath "$raw")" = "$raw" ]
    rm -rf -- "$raw"
    printf '[setup] %s complete frames=%s; temporary AVI removed\n' "$setup" "$actual"
    du -sh "$output"
    df -h "$dataset" | tail -1
done
