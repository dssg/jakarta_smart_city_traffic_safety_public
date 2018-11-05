# ============ Base imports ======================
import os
import subprocess
# ====== External package imports ================
from moviepy.video.io.VideoFileClip import VideoFileClip
# ====== Internal package imports ================
from src.modules.utils.os import syscall_decode as sp
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf, confp = get_config()
# ================================================

raw_dir = confp.dirs.raw
vid_dir = confp.dirs.videos
vid_dir2 = confp.dirs.videos2

def url_to_filename(url):
    """Returns our filename from their url."""

    split_url = url.split('.')[4].split('%')
    part = url.split('.')[5]
    if part[:4] != 'part':
        part = 'part0'
    partial_name = split_url[0] + '_' + split_url[1][2:] + '_' + split_url[2][2:] + '_' + split_url[4][2:]
    file_name = partial_name + '_' + part + '.mkv'
    return file_name

def check_hash(hashed, etag):
    """Checks if two hashes are the same."""
#    logger.info('Checking file: '+file_name)
#    run_hash = subprocess.run('./s3etag.sh %s 7'%(file_name), shell=True, stdout=subprocess.PIPE)
#    hashed = run_hash.stdout.decode('utf-8').replace(' -','').strip()
    return hashed[:32] == etag[:32]

def filelist_to_dict(f):
    """Goes from a file with filenames and hashes to a dict with files as keys and hashes as values."""

    line = f.readline()
    hashes = {}
    while line:
        file_name = line.strip().split('/')[7]
        line = f.readline()
        file_hash = line[:32]
#        print(file_name, file_hash)
        hashes[file_name] = file_hash
        line = f.readline()
    return hashes


def check_files():
    file_list = raw_dir+'/jakarta_data3.csv'
    f = open(file_list, 'r')
    fw = open('files_to_dnld', 'w')

    f_hashes = open('hashes.txt', 'r')
    f_hashes2 = open('hashes2.txt', 'r')

    hashes = filelist_to_dict(f_hashes)
    hashes2 = filelist_to_dict(f_hashes2)
    
    videos = set(os.listdir(vid_dir))
    videos2 = set(os.listdir(vid_dir2))

    corrupted1 = 0
    corrupted2 = 0
    new = 0

    for _ in range(2):
        line = f.readline()
    while line:
#    for _ in range(10):
        cells = line.strip().split(',')
        size = cells[1]
        etag = cells[3]
        url = cells[4]
        
        file_name = url_to_filename(url) 
        wehaveit = False

        if file_name in videos:
            if check_hash(hashes[file_name], etag):
                wehaveit = True
            else:
                corrupted1 += 1
        elif file_name in videos2:
            if check_hash(hashes2[file_name], etag):
                wehaveit = True
            else:
                corrupted2 += 1
        else:
            new += 1
        
        if not wehaveit:
#            logger.info('We don\'t have this file')
            fw.write(url + '\n')
#        else:
#            logger.info('We have this file!')
       
        line = f.readline()

    print('We found %i corrupted files in the first batch, %i in the second batch, and %i new files'%
            (corrupted1, corrupted2, new))

    f_hashes.close()
    f_hashes2.close()
    f.close()
    fw.close()

if __name__=="__main__":
    setup('hash_check')
    check_files()

