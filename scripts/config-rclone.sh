#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SOURCE_DIR=$(dirname "$DIR")

docker run --rm --volume "$SOURCE_DIR/rclone_config":/config/rclone \
    --user $(id -u):$(id -g) \
    -it rclone/rclone:1.54.0 config create data dropbox config_is_local false
