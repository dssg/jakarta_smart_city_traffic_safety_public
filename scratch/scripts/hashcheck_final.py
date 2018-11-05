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
vid_dir3 = confp.dirs.videos3

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
    """Returns our filename from their url."""

#    logger.info('Checking file: '+file_name)
#    run_hash = subprocess.run('./s3etag.sh %s 7'%(file_name), shell=True, stdout=subprocess.PIPE)
#    hashed = run_hash.stdout.decode('utf-8').replace(' -','').strip()
    return hashed[:32] == etag[:32]

def filelist_to_dict(f):
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

    f_hashes = open('hashes.txt', 'r')
    f_hashes2 = open('hashes2.txt', 'r')
    f_hashes3 = open('hashes3.txt', 'r')

    hashes = filelist_to_dict(f_hashes)
    hashes2 = filelist_to_dict(f_hashes2)
    hashes3 = filelist_to_dict(f_hashes3)
    
    videos = set(os.listdir(vid_dir))
    videos2 = set(os.listdir(vid_dir2))
    videos3 = set(os.listdir(vid_dir3))
    
    wehave = 0

    for _ in range(2):
        line = f.readline()
    while line:
#    for _ in range(10):
        cells = line.strip().split(',')
        size = cells[1]
        etag = cells[3]
        url = cells[4]
        
        file_name = url_to_filename(url) 

        if file_name in videos:
            if not check_hash(hashes[file_name], etag):
                logger.info('%s does not match in the first batch:\n%s is our hash, %s is theirs.'%
                        (file_name, hashes[file_name], etag))
                if check_hash(hashes3[file_name], etag):
                    logger.info('Fixed in third batch.')
                    wehave += 1
                else:
                    logger.info('%s is the hash in the third batch: still no match!' % hashes3[file_name])
            else:
                wehave += 1
        elif file_name in videos2:
            if not check_hash(hashes2[file_name], etag):
                logger.info('%s does not match in the second batch:\n%s is our hash, %s is theirs.'%
                        (file_name, hashes2[file_name], etag))
                if check_hash(hashes3[file_name], etag):
                    logger.info('Fixed in third batch.')
                    wehave += 1
                else:
                    logger.info('%s is the hash in the third batch: still no match!' % hashes3[file_name])
            else:
                wehave += 1
        elif file_name in videos3:
            if not check_hash(hashes3[file_name], etag):
                logger.info('%s does not match in the third batch:\n%s is our hash, %s is theirs.'%
                        (file_name, hashes3[file_name], etag))
            else:
                wehave += 1
        else:
            logger.info('We are missing file %s' % file_name)
            
        
        line = f.readline()

    logger.info('We have successfully downloaded %i files.' % wehave)
    f_hashes.close()
    f_hashes2.close()
    f_hashes3.close()
    f.close()

if __name__=="__main__":
    setup('hash_check_final')
    check_files()

