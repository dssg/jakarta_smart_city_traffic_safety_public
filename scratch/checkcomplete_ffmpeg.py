import subprocess
import os

directory_str = "/to_raw/videos/"
directory = os.fsencode(directory_str)

f = open('read_error','w')

for i, filename in enumerate(os.listdir(directory)):
    filename = filename.decode('utf-8')
    if filename[-4:] == '.mkv':
        print(i, filename)
        subpr = ['ffmpeg',
            '-v', 'error',
            '-i', directory_str + filename,
            #'-c:s', 'subrip',
            #'-map', '0:1',
            '-vsync', '0',
            '-f', 'null', '-'
            ] 
        #print(' '.join(subpr))
        run = subprocess.run(subpr, stderr=subprocess.PIPE)
        output = run.stderr.decode('UTF-8')
        #print(output)
        if 'Read error' in output:
            f.write(filename+'\n')

f.close()
