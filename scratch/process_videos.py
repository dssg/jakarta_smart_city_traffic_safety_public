# ============ Base imports ======================
import os
import glob
import hashlib
import gc
# ====== External package imports ================
# ====== Internal package imports ================
from src.modules.data.video_file import VideoFile
from src.modules.data.aws_io import AwsIo
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf, confp = get_config()
# ================================================

vid_dir = confp.dirs.videos
aws = AwsIo()
m = hashlib.md5()


def main():
    for i, f_path in enumerate(glob.glob(vid_dir + "*.mkv")):
        gc.collect()
        logger.info("Processing {}".format(f_path))
        vid = VideoFile(f_path)
        logger.info("Extracting subtitles")
        vid.extract_subtitles()
        logger.info("Extracting Frame statistics")
        vid.extract_frame_stats()
        logger.info("Extracting Packet statistics")
        vid.extract_packet_stats()
        upload = False
        if aws.s3_vid_exists(vid.basename):
            logger.info("Video already exists on aws")
            if aws.s3_get_vid_size(vid.basename) == os.path.getsize((os.path.join(vid_dir, vid.basename))):
                logger.info("Sizes match")
            else:
                logger.info("Sizes don't match")
                upload = True
        else:
            logger.info("Video not already on aws")
            upload = True
        if upload:
            logger.info("Uploading")
            aws.vid_copy_file_to_s3(vid.path)
            logger.info("Done")
        else:
            logger.info("Not uploading")


if __name__=="__main__":
    setup("process_videos")
    main()
