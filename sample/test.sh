#!/usr/bin/env bash
python ../trakingfilter.py |tee log.txt


# Convert o.avi to o.mp4 by ffmpeg:
#  ffmpeg -i o.avi -f mp4 -vcodec h264 -qscale 20 -acodec aac -ab 128 o.mp4
