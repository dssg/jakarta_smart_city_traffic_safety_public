#!/bin/sh

vid_dir=$1
out_dir=$2

echo $vid_dir
count=0
for vid in `find $vid_dir -name '*.mkv'` ; do
    count=$((count + 1))
    echo "Running ffmpeg on file $count: $vid"
    # get filename
    filename=$(basename $vid ".mkv")
    ffmpeg -i $vid -map 0:0 "$out_dir/$filename.srt"
done
