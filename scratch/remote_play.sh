user=$1
host=$2
file=$3

ssh $user@$host cat $file | mplayer -
