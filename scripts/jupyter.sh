#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SOURCE_DIR=$(dirname "$DIR")

exec docker run --runtime=nvidia \
                --rm \
                --interactive \
                --tty \
                --publish 8888:8888 \
                --publish 6006:6006 \
                --user $(id -u):$(id -g) \
                --mount type=bind,source="$SOURCE_DIR",target=/code \
                english-french-nmt \
                /run_jupyter.sh --allow-root --ip=0.0.0.0
