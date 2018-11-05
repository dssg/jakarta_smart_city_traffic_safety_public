$!/bin/sh

vid=$1
output=$2

ffprobe -loglevel panic -of csv -select_streams v -show_frames $vid > $output
exit
