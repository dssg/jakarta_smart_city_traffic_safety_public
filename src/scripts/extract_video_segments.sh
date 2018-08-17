#!/bin/bash

# get the videos location
vid_dir=$1

# samples file
samples_file=$2

# get the output location
output_dir=$3

#vid_dir="/mnt/data/projects/jakarta_smart_city_traffic_safety/raw/videos/"
#samples_file="/mnt/data/projects/jakarta_smart_city_traffic_safety/output/video_sample.csv"
#output_dir="/mnt/data/projects/jakarta_smart_city_traffic_safety/data_processed/video_samples/"



firstline=true

# loop through each video in file
while read line; do
    if $firstline ; then
        firstline=false
    else
        IFS=',' read -ra split_line <<< "$line"
        IFS='.' read -ra tstart <<< "${split_line[4]}"
        IFS='.' read -ra tend <<< "${split_line[5]}"
	    tstart=$(($tstart+1))
        infilename="$vid_dir${split_line[0]}.mkv"
        outfilename="$output_dir${split_line[0]}_"$tstart"_to_"$tend".mkv"
        echo "executing:ffmpeg -nostdin -i $infilename -ss $tstart -to $tend -c copy $outfilename"
        ffmpeg -nostdin -i $infilename -ss $tstart -to $tend -c copy $outfilename
    fi
done < $samples_file
