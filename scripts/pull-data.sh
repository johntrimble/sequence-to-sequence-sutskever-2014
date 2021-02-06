#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SOURCE_DIR=$(dirname "$DIR")

for directory in "downloads" "model_checkpoints"; do
    docker run --rm \
        --volume "$SOURCE_DIR":/data \
        --volume "$SOURCE_DIR/rclone_config":/config/rclone \
        --user $(id -u):$(id -g) \
        --tty \
        --interactive \
        rclone/rclone:1.54.0 \
        copy --progress \
        --checksum \
        "data:Data/sequence-to-sequence-sutskever-2014/$directory" "/data/$directory"
done
