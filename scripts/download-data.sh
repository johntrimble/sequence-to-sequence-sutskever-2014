#!/usr/bin/env bash
set -e

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

DOWNLOADS="${BASE_DIR}/downloads"
TARGET="${BASE_DIR}/target"

# http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/
BITEXTS_URL='http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/data/bitexts.tgz'
BITEXTS_PATH="${DOWNLOADS}/bitexts.tgz"

DEV_TEST_URL='http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/data/dev+test.tgz'
DEV_TEST_PATH="${DOWNLOADS}/dev+test.tgz"

function download() {
  local url=$1
  local download_path=$2
  if [ ! -f "$download_path" ]; then
    wget --continue --output-document="$download_path" "$url"
  fi
}

function extract() {
  local path="$1"
  if [ ${path: -4} == ".zip" ]; then
    unzip "$path" -d "$TARGET"
  else
    tar -xzf "$path" --directory "$TARGET"
  fi
}

# create directories
mkdir -p "$DOWNLOADS"
mkdir -p "$TARGET"

download "$BITEXTS_URL" "$BITEXTS_PATH"
download "$DEV_TEST_URL" "$DEV_TEST_PATH"
extract "$BITEXTS_PATH"
extract "$DEV_TEST_PATH"
