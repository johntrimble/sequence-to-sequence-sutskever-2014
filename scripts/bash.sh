#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SOURCE_DIR="$(dirname "$DIR")"

exec docker run --runtime=nvidia \
                --rm \
                --interactive \
                --tty \
                --user $(id -u):$(id -g) \
                --mount type=bind,source="$SOURCE_DIR",target=/code \
                english-french-nmt \
                /bin/bash
