# ============ Base imports ======================
import os
import requests
import datetime
import schedule
import time
import yaml
# ====== External package imports ================
from src.modules.utils.misc import run_and_catch_exceptions
# ====== Internal package imports ================
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


def download_file(cctv, min_back,unix_timestamp):
    """Downloads video files from urls

    for each url in cctv, downloads a video file starting min_back minutes before unix_timestamp and ending at
    unix_timestamp. places videos in conf.dirs.downloaded_videos

    :param cctv: list of camera urls
    :param min_back: number of minutes before unix_timestamp to start the video
    :param unix_timestamp: end of video segment.
    :return: time that this job finished
    """
    ###Create directory if !exist
    logger.info(f"Starting downloads")
    if not os.path.exists(conf.dirs.downloaded_videos):
        os.makedirs(conf.dirs.downloaded_videos)

    for vid in cctv:
        logger.info(f"Downloading vid: {vid}")
        sec=min_back * 60 # e.g. 5 min * 60 seconds
        unix_time = int(unix_timestamp - (sec))
        u_time= str(unix_time) + "-" + str(sec) + ".mp4"
        
        url= vid + "/archive-" + u_time 

        ###download video
        local_filename = os.path.join(conf.dirs.downloaded_videos, vid.split(".co.id/")[1] + "-" + u_time)

        r = requests.get(url, stream=True)

        file_size = 0
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024): 
                if chunk: # filter out keep-alive new chunks
                    file_size += 1024
                    f.write(chunk)
        logger.info(f"Done, file size {file_size}")

    return datetime.datetime.now()


def job():
    """gets current unix time and calls the download_file function, getting last 5 minutes of footage

    """
    ###func takes in list and duration of video in minutes
    with open(conf.files.camera_urls) as f:
        cctv = yaml.load(f.read())
    current_time = datetime.datetime.now(datetime.timezone.utc)
    unix_timestamp = current_time.timestamp()
    logger.info(download_file(cctv,5,unix_timestamp))


def main():
    """Runs a job every 5 minutes which downloads last 5 minutes of footage from cameras.

    """
    # Get list of cameras
    schedule.every(5).minutes.do(job)
    while 1:
        logger.info("Pausing 5 minutes...")
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    script_name = os.path.basename(__file__).split(".")[0]
    setup(script_name)
    run_and_catch_exceptions(logger, main)
