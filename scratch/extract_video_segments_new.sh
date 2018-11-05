#!/bin/bash

# get the videos location
vid_dir=$1

# samples file
samples_file=$2

# get the output location
output_dir=$3

vid_dir="/scratch/scripts/new_videos/"
samples_file="/output/video_sample_new.csv"
output_dir="/data_processed/video_samples/"


firstline=true

# loop through each video in file
while read line; do
    if $firstline ; then
        firstline=false
    else
        IFS=',' read -ra split_line <<< "$line"
        IFS='.' read -ra tstart <<< "${split_line[4]}"
        IFS='.' read -ra tend <<< "${split_line[5]}"
        infilename="$vid_dir${split_line[0]}.mp4"
        outfilename="$output_dir${split_line[0]}_"$tstart"_to_"$tend".mp4"
        echo "executing:ffmpeg -nostdin -i $infilename -ss $tstart -to $tend $outfilename"
        echo ""
        ffmpeg -nostdin -i $infilename -ss $tstart -to $tend $outfilename
        echo ""
    fi
done < $samples_file
