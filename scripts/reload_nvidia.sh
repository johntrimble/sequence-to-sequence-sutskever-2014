#!/usr/bin/env bash

# https://github.com/tensorflow/tensorflow/issues/5777
# sudo rmmod nvidia_uvm
# sudo rmmod nvidia_drm
# sudo rmmod nvidia_modeset
# sudo rmmod nvidia
# sudo modprobe nvidia
# sudo modprobe nvidia_modeset
# sudo modprobe nvidia_drm
# sudo modprobe nvidia_uvm

nvidia-smi pmon -c 1 | awk '$3 == "C" { print $2 }' | xargs -I% kill %
