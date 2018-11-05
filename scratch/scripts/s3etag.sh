#!/bin/bash

# Error handling to set part of file size in MB correctly
if [ $# -ne 2 ]; then
    echo "Usage: $0 file partSizeInMb";
    exit 0;
fi

file=$1


# Error handling to inform file is not found
if [ ! -f "$file" ]; then
    echo "Error: $file not found." 
    exit 1;
fi

# Variable partSizeinMb is set for second parameter
# Variable fileSizeInMb is gotten by calculate disk usage of file
partSizeInMb=$2
fileSizeInMb=$(du -m "$file" | cut -f 1)


# Run these code if file size is lower than 7 MB
# And automatically hash file using MD5
if [ $fileSizeInMb -lt 7 ]; then
	md5sum $file
	exit 2;
fi

# Run these code if file size is greater than 7 MB
# Chunk file into parts
parts=$((fileSizeInMb / partSizeInMb))
if [[ $((fileSizeInMb % partSizeInMb)) -gt 0 ]]; then
    parts=$((parts + 1));
fi

checksumFile=$(mktemp -t s3md5.XXX)

for (( part=0; part<$parts; part++ ))
do
    skip=$((partSizeInMb * part))
    $(dd bs=1M count=$partSizeInMb skip=$skip if=$file 2>/dev/null | md5sum >>$checksumFile)
done

echo $(xxd -r -p $checksumFile | md5sum)-$parts
rm $checksumFile
