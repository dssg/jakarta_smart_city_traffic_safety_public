# ============ Base imports ======================
import os
import urllib.request
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

vid_dir = confp.dirs.videos3

def url_to_filename(url):
    split_url = url.split('.')[4].split('%')
    part = url.split('.')[5]
    if part[:4] != 'part':
        part = 'part0'
    partial_name = split_url[0] + '_' + split_url[1][2:] + '_' + split_url[2][2:] + '_' + split_url[4][2:]
    file_name = partial_name + '_' + part + '.mkv'
    return file_name

def download():
    f = open('files_to_dnld')
    url = f.readline().strip()
    while url:
        # get just a few bits for file naming
        file_name = url_to_filename(url)
        # check if file already exists
        full_path = vid_dir + '/' + file_name
        if os.path.isfile(full_path):
            logger.info("skipping" + file_name)
            url = f.readline().strip()
            continue

        logger.info(file_name)
        urllib.request.urlretrieve(url, full_path)
        
        url = f.readline().strip()

    f.close()
    return(True)       

if __name__=="__main__":
    setup('download')
    download()

