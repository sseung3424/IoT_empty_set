#!/usr/bin/env bash
# Simple H.264 TCP stream on port 8888 (low-latency)
rpicam-vid --inline --codec h264 --width 640 --height 480 --framerate 30 \
  --listen -o tcp://0.0.0.0:8888
